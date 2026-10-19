#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash dense-weight inventory per TP rank: which dtype/scale
format each dense weight has in LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4,
the per-rank shape under TP8 (+EP), bytes, the SM120 kernel vLLM
39983a5e7 runs today, and a bandwidth/launch roofline floor at decode M.

    python3 scripts/dsv41_dense_inventory.py                 # markdown tables
    python3 scripts/dsv41_dense_inventory.py --json          # machine form
    python3 scripts/dsv41_dense_inventory.py --verify        # check the CKPT rows against the
                                                             # real safetensors headers (HTTP range)

Every number below ROW_* is read from the checkpoint headers (--verify
re-checks them); every time is an ESTIMATE from the assumed constants, not a
measurement (bytes / achievable bandwidth + launch gaps; the compute floor
uses an assumed effective tensor-core rate). Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys

HBM_GBS = 1597.0      # RTX PRO 6000 Blackwell Server Edition, spec sheet (assumption: nominal)
BW_EFF = 0.85         # achievable fraction for a well-pipelined streaming kernel (assumption)
LAUNCH_US = 2.0       # CUDA-graph kernel-to-kernel gap per extra launch (assumption)
SPLIT_EXTRA_LAUNCH = 2  # two-stage split-K: torch sum + bf16 copy after the Helion kernel (capture-safe default)
FP8_TFLOPS = 400.0    # effective dense FP8 (e4m3 x e4m3, fp32 acc) mma rate (assumption)
BF16_TFLOPS = 200.0   # effective dense BF16 mma rate (assumption)
MS = (1, 6, 48, 192)  # decode rows: 1 (no spec) / 6 (c1 DSpark) / 48 (8 seqs) / 192 (32 seqs)

L, D = 40, 3          # backbone layers / DSpark draft layers (mtp.0-2, dense shapes = backbone)
H, QL, OG, OL = 5120, 1280, 8, 1024

# kind: fp8 = MXFP8 (e4m3 + per-32 ue8m0, vLLM expands the ckpt's 32x32 scale blocks per row),
#       bf16 / f32 = unquantized
# stock: the SM120 kernel today; ours: lane (None = no lane / not planned)
# n = invocations per decode step per rank; launches = kernel launches per invocation
# ckpt = (tensor key, dtype, shape) used by --verify (layer 0 unless noted)


def ceil_div(a, b):
    return -(-a // b)


def rows(tp: int = 8) -> list[dict]:
    inter = ceil_div(2304 // 32, tp) * 32  # shared expert I per rank, whole 32-blocks
    return [
        dict(name="fused_wqa_wkv", kind="fp8", N=QL + 512, K=H, n=L + D, tp="replicated",
             stock="FlashInfer CUTLASS mm_mxfp8 + dynamic MXFP8 act quant", stock_launch=2,
             ours="mxfp8_gemm A16/A8 (SUFFIX_HELION_MXFP8_GEMM)", ours_launch=1,
             ckpt=[("layers.0.attn.wq_a.weight", "F8_E4M3", [QL, H]),
                   ("layers.0.attn.wkv.weight", "F8_E4M3", [512, H])]),
        dict(name="wq_b", kind="fp8", N=64 * 512 // tp, K=QL, n=L + D, tp="column",
             stock="FlashInfer CUTLASS mm_mxfp8 (act pre-quantized by the fused q-norm)",
             stock_launch=1, ours="mxfp8_gemm A16/A8", ours_launch=1,
             ckpt=[("layers.0.attn.wq_b.weight", "F8_E4M3", [32768, QL])]),
        dict(name="indexer.wq_b", kind="fp8", N=4096, K=QL, n=8, tp="replicated",
             stock="FlashInfer CUTLASS mm_mxfp8 (pre-quantized act)", stock_launch=1,
             ours="mxfp8_gemm A16/A8", ours_launch=1,
             ckpt=[("layers.2.attn.indexer.wq_b.weight", "F8_E4M3", [4096, QL])]),
        dict(name="wo_a (grouped o_proj)", kind="fp8", N=OG // tp * OL, K=64 * 512 // OG, n=L + D,
             tp="column (1 group/rank)",
             stock="EmulationMxfp8: BF16 weight dequantized AT LOAD + torch.bmm", stock_bf16=True,
             stock_launch=1, ours="mxfp8_gemm A16 on the FP8 weight (wo_a lane)", ours_launch=1,
             ckpt=[("layers.0.attn.wo_a.weight", "F8_E4M3", [OG * OL, 64 * 512 // OG])]),
        dict(name="wo_b", kind="fp8", N=H, K=OG * OL // tp, n=L + D, tp="row",
             stock="FlashInfer CUTLASS mm_mxfp8 + dynamic act quant", stock_launch=2,
             ours="mxfp8_gemm A16/A8", ours_launch=1,
             ckpt=[("layers.0.attn.wo_b.weight", "F8_E4M3", [H, OG * OL])]),
        dict(name="shared.gate_up (w1+w3)", kind="fp8", N=2 * inter, K=H, n=L + D, tp="column (padded)",
             stock="FlashInfer CUTLASS mm_mxfp8 + dynamic act quant", stock_launch=2,
             ours="mxfp8_gemm A16/A8", ours_launch=1,
             ckpt=[("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", [2304, H]),
                   ("layers.0.ffn.shared_experts.w3.weight", "F8_E4M3", [2304, H])]),
        dict(name="shared.down (w2)", kind="fp8", N=H, K=inter, n=L + D, tp="row (padded)",
             stock="FlashInfer CUTLASS mm_mxfp8 + dynamic act quant", stock_launch=2,
             ours="mxfp8_gemm A16/A8", ours_launch=1,
             ckpt=[("layers.0.ffn.shared_experts.w2.weight", "F8_E4M3", [H, 2304])]),
        dict(name="dspark.main_proj", kind="fp8", N=H, K=3 * H, n=1, tp="replicated",
             stock="FlashInfer CUTLASS mm_mxfp8 + dynamic act quant", stock_launch=2,
             ours="mxfp8_gemm A16/A8", ours_launch=1,
             ckpt=[("mtp.0.main_proj.weight", "F8_E4M3", [H, 3 * H])]),
        dict(name="engram.wkv", kind="fp8", N=25600 // tp, K=6144, n=2, tp="column (gather)",
             stock="FlashInfer CUTLASS mm_mxfp8 on the FP8 table-prefetch activation",
             stock_launch=1, ours=None, ours_launch=1,
             note="input arrives pre-quantized (swizzled scales): lane excludes `.engram.`",
             ckpt=[("layers.1.engram.wkv.weight", "F8_E4M3", [25600, 6144])]),
        # --- unquantized leftovers (vLLM runs them as plain BF16 / F32) ---
        dict(name="lm_head", kind="bf16", N=129280 // tp, K=H, n=2, tp="vocab-parallel",
             stock="cuBLAS BF16 GEMM (UnquantizedEmbeddingMethod); n=2: target + drafter base logits",
             stock_launch=1, ours="NVFP4 lm_head lane (SUFFIX_NVFP4_LMHEAD, oxide, M<=64)",
             ours_launch=1, ours_bytes=0.5625, ours_max_m=64,
             ckpt=[("head.weight", "BF16", [129280, H])]),
        dict(name="dspark markov_w2 (dense path)", kind="bf16", N=129280, K=256, n=5,
             tp="replicated (disable_tp)", per_seq=True,
             stock="cuBLAS BF16 GEMM x5 sequential (dspark_draft_topk unset -> markov_bias)",
             stock_launch=1, ours=None, ours_launch=1,
             note="draft-side only: requant needs an acceptance/tok-s gate, not a KL gate",
             ckpt=[("mtp.2.markov_head.head.weight", "BF16", [129280, 256])]),
        dict(name="router gate", kind="bf16", N=384, K=H, n=L, tp="replicated",
             stock="cuBLAS BF16 (+ DSpark 3 x [128, K])", stock_launch=1, ours=None,
             ours_launch=1, note="MoE-routing lane's; listed for the byte budget",
             ckpt=[("layers.0.ffn.gate.weight", "BF16", [384, H])]),
        dict(name="hc_{attn,ffn}_fn (mHC mixing)", kind="f32", N=2 * 24, K=4 * H, n=L + D,
             tp="replicated", stock="mHC tilelang/Triton kernels (not a Linear)", stock_launch=1,
             ours=None, ours_launch=1, note="mHC lane's; listed for the byte budget",
             ckpt=[("layers.0.hc_attn_fn", "F32", [24, 4 * H])]),
        dict(name="compressor wkv (kv-source layers)", kind="bf16", N=512, K=H, n=4,
             tp="replicated", stock="cuBLAS BF16", stock_launch=1, ours=None, ours_launch=1,
             ckpt=[("layers.2.attn.compressor.wkv.weight", "BF16", [512, H])]),
        dict(name="compressor wgate", kind="bf16", N=512, K=H, n=3, tp="replicated",
             stock="cuBLAS BF16", stock_launch=1, ours=None, ours_launch=1,
             ckpt=[("layers.2.attn.compressor.wgate.weight", "BF16", [512, H])]),
        dict(name="indexer weights_proj", kind="bf16", N=32, K=H, n=8, tp="replicated",
             stock="cuBLAS BF16 gemv", stock_launch=1, ours=None, ours_launch=1,
             ckpt=[("layers.2.attn.indexer.weights_proj.weight", "BF16", [32, H])]),
    ]


def weight_bytes(r: dict, native: bool = True) -> float:
    """HBM bytes one invocation streams: FP8 = N*K + N*K/32 scales (per-row expanded);
    stock wo_a = BF16 (2 B); bf16 = 2 B; f32 = 4 B; ours_bytes = factor per weight elem."""
    nk = r["N"] * r["K"]
    if r["kind"] == "fp8":
        return 2.0 * nk if (not native and r.get("stock_bf16")) else nk * (1 + 1 / 32)
    return nk * (2.0 if r["kind"] == "bf16" else 4.0)


def m_of(r: dict, m: int) -> int:
    """GEMM rows for a step of m verified tokens: the DSpark Markov head runs once per
    SEQUENCE and draft position (m / 6 sequences)."""
    return max(1, m // 6) if r.get("per_seq") else m


def call_us(r: dict, m: int, bytes_: float, launches: int, fp8: bool) -> float:
    m = m_of(r, m)
    bw = HBM_GBS * 1e9 * BW_EFF
    peak = (FP8_TFLOPS if fp8 else BF16_TFLOPS) * 1e12
    act = m * r["K"] * 2 + m * r["N"] * 2  # x in (bf16) + y out
    return max((bytes_ + act) / bw, 2.0 * m * r["N"] * r["K"] / peak) * 1e6 + (launches - 1) * LAUNCH_US


def table(tp: int = 8) -> list[dict]:
    out = []
    for r in rows(tp):
        sb = weight_bytes(r, native=False)  # what stock streams
        ob = (r["N"] * r["K"] * r["ours_bytes"]) if r.get("ours_bytes") else weight_bytes(r)
        e = dict(name=r["name"], kind=r["kind"], N=r["N"], K=r["K"], n=r["n"], tp=r["tp"],
                 stock=r["stock"], ours=r["ours"], note=r.get("note", ""),
                 stock_MB=sb / 1e6, ours_MB=ob / 1e6 if r["ours"] else None,
                 step_stock_MB=sb * r["n"] / 1e6,
                 step_ours_MB=(ob * r["n"] / 1e6) if r["ours"] else None, us={})
        for m in MS:
            s = call_us(r, m, sb, r["stock_launch"], r["kind"] == "fp8" and not r.get("stock_bf16"))
            o = (call_us(r, m, ob, r["ours_launch"], r["kind"] == "fp8" and not r.get("stock_bf16"))
                 if r["ours"] and m <= r.get("ours_max_m", 1 << 30) else s if r["ours"] else None)
            # the same lane when the tuned config splits K: + the two reduce launches
            sp = (call_us(r, m, ob, r["ours_launch"] + SPLIT_EXTRA_LAUNCH,
                          r["kind"] == "fp8" and not r.get("stock_bf16"))
                  if r["ours"] and m <= r.get("ours_max_m", 1 << 30) else o)
            e["us"][m] = dict(stock_step=s * r["n"], ours_step=None if o is None else o * r["n"],
                              ours_split_step=None if sp is None else sp * r["n"])
        out.append(e)
    return out


def markdown(tp: int = 8) -> str:
    t = table(tp)
    tot = {m: sum(e["us"][m]["stock_step"] for e in t) for m in MS}
    out = [f"### Per-rank decode inventory, TP{tp} + EP (ESTIMATE: {HBM_GBS:.0f} GB/s x {BW_EFF}, "
           f"{LAUNCH_US} us/extra launch, {FP8_TFLOPS:.0f}/{BF16_TFLOPS:.0f} TFLOPS fp8/bf16 effective)", "",
           "| component | dtype | N x K | calls/step | MB/call stock | MB/step stock | "
           + " | ".join(f"stock us/step M={m}" for m in MS) + " | share M=6 |",
           "|---|---|---|---|---|---|" + "---|" * (len(MS) + 1)]
    for e in sorted(t, key=lambda e: -e["us"][6]["stock_step"]):
        out.append(f"| {e['name']} | {e['kind']} | {e['N']} x {e['K']} | {e['n']} | {e['stock_MB']:.2f} | "
                   f"{e['step_stock_MB']:.0f} | " + " | ".join(f"{e['us'][m]['stock_step']:.0f}" for m in MS)
                   + f" | {100 * e['us'][6]['stock_step'] / tot[6]:.1f}% |")
    out.append(f"| **total** | | | | | {sum(e['step_stock_MB'] for e in t):.0f} | "
               + " | ".join(f"**{tot[m]:.0f}**" for m in MS) + " | |")
    out += ["", "### Where our lane changes the bytes/launches (same constants)", "",
            "Single launch = the config does not split K (stage 1 only). Split-K = the capture-safe two-stage "
            f"default: + {SPLIT_EXTRA_LAUNCH} reduce launches after the kernel (what a tuned split costs).", "",
            "| component | stock MB/step | ours MB/step | "
            + " | ".join(f"saved us/step M={m}" for m in MS) + " | saved us/step M=6 if split-K |",
            "|---|---|---|" + "---|" * (len(MS) + 1)]
    for e in t:
        if e["ours"] is None:
            continue
        out.append(f"| {e['name']} | {e['step_stock_MB']:.0f} | {e['step_ours_MB']:.0f} | "
                   + " | ".join(f"{e['us'][m]['stock_step'] - e['us'][m]['ours_step']:.0f}" for m in MS)
                   + f" | {e['us'][6]['stock_step'] - e['us'][6]['ours_split_step']:.0f} |")
    mx = [e for e in t if e["ours"] and e["kind"] == "fp8"]
    out.append("| **MXFP8 lane subtotal** | "
               f"{sum(e['step_stock_MB'] for e in mx):.0f} | {sum(e['step_ours_MB'] for e in mx):.0f} | "
               + " | ".join(f"**{sum(e['us'][m]['stock_step'] - e['us'][m]['ours_step'] for e in mx):.0f}**"
                            for m in MS)
               + f" | **{sum(e['us'][6]['stock_step'] - e['us'][6]['ours_split_step'] for e in mx):.0f}** |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# --verify: the CKPT rows against the real safetensors headers (HTTP range)
# ---------------------------------------------------------------------------
REPO = "LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4"


def _get(url: str, rng: str | None = None) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (fixed https HF url)
        return r.read()


def verify() -> int:
    import struct
    base = f"https://huggingface.co/{REPO}/resolve/main/"
    wm = json.loads(_get(base + "model.safetensors.index.json"))["weight_map"]
    cache, bad = {}, 0
    for r in rows():
        for key, dtype, shape in r.get("ckpt", []):
            f = wm[key]
            if f not in cache:
                n = struct.unpack("<Q", _get(base + f, "0-7"))[0]
                cache[f] = json.loads(_get(base + f, f"8-{8 + n - 1}"))
            h = cache[f][key]
            ok = h["dtype"] == dtype and h["shape"] == shape
            bad += not ok
            print(f"{'OK  ' if ok else 'FAIL'} {key:55s} {h['dtype']:7s} {h['shape']}"
                  + ("" if ok else f"   expected {dtype} {shape}"))
    print("all ckpt rows match" if not bad else f"{bad} MISMATCH")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return verify()
    print(json.dumps(table(a.tp), indent=1) if a.json else markdown(a.tp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
