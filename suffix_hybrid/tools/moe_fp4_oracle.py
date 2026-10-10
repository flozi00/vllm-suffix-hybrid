# SPDX-License-Identifier: Apache-2.0
"""Numerics oracle: aiter.fused_moe default heuristic vs tuned CSV row on vLLM-style
padded Quark MXFP4 W4A4 MoE weights (inter 640 -> 768, E=513 incl. fused shared expert,
top-11, hidden 2560). Boot gate moe_fp4_oracle runs it with SUFFIX_ROCM_AITER_FLYDSL_PAD=1 (rocm_patches
then rewrites aiter.fused_moe on import), so every case must PASS:

    python -m suffix_hybrid.tools.moe_fp4_oracle [tuned_fmoe.csv]   # default: the bundled table

torch.empty is poisoned with 0xFF during the MoE call (E8M0 0xFF = NaN, bf16 0xFFFF = NaN),
so any kernel that consumes memory it never wrote shows up deterministically instead of
depending on stale allocator contents.
Expected on unpatched AITER v0.1.24.post1: "tuned pad=128" FAIL (NaN rows), "tuned pad=0" PASS.
After applying patches/aiter-v0.1.24.post1-flydsl-layout-stage1-pad0.patch: all PASS.
"""
import os
import sys

_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                    "mi350p_tuned_fmoe.csv")
if __name__ == "__main__" and len(sys.argv) > 1:
    os.environ["AITER_CONFIG_FMOE"] = sys.argv[1]  # must be set before importing aiter
elif __name__ == "__main__":
    os.environ["AITER_CONFIG_FMOE"] = _CSV
# Imported (moe_flat_probe): the importer set AITER_CONFIG_FMOE already.

import torch  # noqa: E402

import aiter.fused_moe as fm  # noqa: E402
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.ops.quant import per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility.fp4_utils import e8m0_shuffle  # noqa: E402

E, TOPK, H, I, IP = 513, 11, 2560, 640, 768  # expert 512 = fused shared expert
dev = "cuda"
torch.manual_seed(0)


def quant(w):  # [n, N, K] bf16 -> (fp4 bytes [n, N, K/2], e8m0 bytes [n, N, K/32])
    y, s = per_1x32_f4_quant(w.reshape(-1, w.shape[-1]))
    n, N = w.shape[:2]
    return y.view(torch.uint8).view(n, N, -1), s.view(torch.uint8).view(n, N, -1)


# Same padded layout vLLM builds (quark_moe.create_weights: zeros weights, scale byte 1;
# routed_experts._load_w13/_load_w2 narrow): gate rows [0,640), up rows [768,1408), w2 K [0,640).
w13 = torch.zeros(E, 2 * IP, H // 2, dtype=torch.uint8, device=dev)
w13s = torch.ones(E, 2 * IP, H // 32, dtype=torch.uint8, device=dev)
w2 = torch.zeros(E, H, IP // 2, dtype=torch.uint8, device=dev)
w2s = torch.ones(E, H, IP // 32, dtype=torch.uint8, device=dev)
for e0 in range(0, E, 32):
    e1 = min(E, e0 + 32)
    g, gs = quant(torch.randn(e1 - e0, 2 * I, H, device=dev, dtype=torch.bfloat16) / H**0.5)
    w13[e0:e1, :I], w13s[e0:e1, :I] = g[:, :I], gs[:, :I]
    w13[e0:e1, IP : IP + I], w13s[e0:e1, IP : IP + I] = g[:, I:], gs[:, I:]
    d, ds = quant(torch.randn(e1 - e0, H, I, device=dev, dtype=torch.bfloat16) / I**0.5)
    w2[e0:e1, :, : I // 2], w2s[e0:e1, :, : I // 32] = d, ds
# oracle/mxfp4.py AITER_MXFP4_MXFP4: e8m0_shuffle scales + shuffle_weight((16, 16))
W1 = shuffle_weight(w13.view(dtypes.fp4x2), (16, 16))
W2 = shuffle_weight(w2.view(dtypes.fp4x2), (16, 16))
S1 = e8m0_shuffle(w13s.view(E * 2 * IP, -1)).view(E, 2 * IP, -1)
S2 = e8m0_shuffle(w2s.view(E * H, -1)).view(E, H, -1)

_empty = torch.empty


def _poisoned_empty(*a, **k):
    t = _empty(*a, **k)
    if t.is_cuda and t.numel():
        t.view(-1).view(torch.uint8).fill_(0xFF)
    return t


picked = []
_cfgs = fm.get_2stage_cfgs


def _spy(*a, **k):
    md = _cfgs(*a, **k)
    kn = [getattr(s, "keywords", {}).get("kernelName") for s in (md.stage1, md.stage2)]
    picked.append(kn)
    return md


fm.get_2stage_cfgs = _spy


def run(x, w, ids, pad, tuned):
    os.environ["AITER_BYPASS_TUNE_CONFIG"] = "0" if tuned else "1"
    _cfgs.cache_clear()
    picked.clear()
    torch.empty = _poisoned_empty
    try:
        out = fm.fused_moe(
            x, W1, W2, w, ids, None, ActivationType.Silu, QuantType.per_1x32, False,
            S1, S2, None, None, dtype=torch.bfloat16, hidden_pad=0,
            intermediate_pad=pad, swiglu_limit=0.0,
        )
        torch.cuda.synchronize()
    finally:
        torch.empty = _empty
    return out.float(), picked[0]  # [0] = the row _fused_moe_impl actually resolved


def main() -> int:
    ok = True
    inputs, first = {}, {}
    for M in (8, 32, 128, 256):
        x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E - 1, device=dev)[: TOPK - 1] for _ in range(M)])
        ids = torch.cat([ids, torch.full((M, 1), E - 1, device=dev)], 1).to(torch.int32)
        w = torch.cat(
            [torch.softmax(torch.randn(M, TOPK - 1, device=dev), -1),
             torch.sigmoid(torch.randn(M, 1, device=dev))], 1
        ).float()
        inputs[M] = x, w, ids
        ref, kref = run(x, w, ids, 128, tuned=False)  # prod default path (GSM8K 0.97)
        print(f"[suffix moe-fp4-oracle] M={M} default pad=128: nan_rows={int(ref.isnan().any(1).sum())} {kref}")
        for name, pad, tuned in (("tuned pad=128", 128, True), ("tuned pad=0", 0, True),
                                 ("default pad=0", 0, False)):
            out, kn = run(x, w, ids, pad, tuned)
            nan_rows = int(out.isnan().any(1).sum())
            cos = torch.nn.functional.cosine_similarity(
                out.nan_to_num().flatten(), ref.nan_to_num().flatten(), 0).item()
            good = nan_rows == 0 and cos >= 0.99
            ok &= good
            first.setdefault((M, name), out)
            print(f"[suffix moe-fp4-oracle]   {name:14s} {'PASS' if good else 'FAIL'} nan_rows={nan_rows} cos={cos:.5f} {kn}")
    # Largest first: buffers that outlive a call (SUFFIX_ROCM_AITER_FLYDSL_ZBUF) now hold a larger
    # call's data wherever the smaller one does not write; a padded tail that moved with the size
    # would read it.
    for M in (256, 128, 32, 8):
        x, w, ids = inputs[M]
        out, kn = run(x, w, ids, 128, True)
        nan_rows = int(out.isnan().any(1).sum())
        cos = torch.nn.functional.cosine_similarity(
            out.nan_to_num().flatten(), first[(M, "tuned pad=128")].flatten(), 0).item()
        good = nan_rows == 0 and cos >= 0.9999
        ok &= good
        print(f"[suffix moe-fp4-oracle] M={M} tuned pad=128 after larger calls: "
              f"{'PASS' if good else 'FAIL'} nan_rows={nan_rows} cos vs first call={cos:.6f}")
    print("[suffix moe-fp4-oracle] " + ("ALL PASS" if ok else "SOME FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
