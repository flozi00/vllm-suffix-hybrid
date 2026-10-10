# SPDX-License-Identifier: Apache-2.0
"""AITER's FLAT one-stage asm MoE at Qwen3.8-Flash-Next decode sizes (boot gate moe_flat_probe).

AITER v0.1.24.post1 runs fmoe_bf16_pertokenMXfp4_g1u1_flat_* only from a tuned row
(run_1stage = xbf16 = flat = 1): raw top-k in, MXFP4 activation quant inside the kernel, one
launch instead of two sorts + quant/sort + stage 1 + stage 2 (+ reduction). AMD tunes it for
Qwen3.8-2400B (H 8192 / I 256) at 1..16 tokens; this checks our H 2560 / I 768 (vLLM-padded
640), E 513 / top-11 (fused shared expert): per FLAT kernel, a CSV = the shipped table with
FLAT rows for tokens 1..16, run in a child process (AITER reads the table at import): NaN rows
and cosine vs AITER's default path (moe_fp4_oracle.run: torch.empty poisoned), then graphed
us/call next to the shipped table's FlyDSL rows (child "shipped").

    python -m suffix_hybrid.tools.moe_flat_probe
"""
import csv
import os
import subprocess
import sys
import tempfile

KERNELS = {
    "vs16x32": "_ZN5aiter47fmoe_bf16_pertokenMXfp4_g1u1_flat_vs_silu_16x32E",
    "novs16x128": "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x128E",
    "novs16x256": "_ZN5aiter50fmoe_bf16_pertokenMXfp4_g1u1_flat_novs_silu_16x256E",
}
MS = (1, 2, 4, 5, 8, 16)  # draft passes (1), c1 verify (5), small batches
SHIPPED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                       "mi350p_tuned_fmoe.csv")
MARK = "[suffix moe-flat]"


def flat_csv(kernel: str) -> str:
    """The shipped table with tokens 1..16 (E 513 / top-11 and E 512 / top-10) as FLAT rows."""
    with open(SHIPPED) as f:
        rows = list(csv.DictReader(f))
    cols = list(rows[0])
    for expert, topk in (("513", "11"), ("512", "10")):
        base = next(r for r in rows if r["expert"] == expert)
        for t in ("1", "2", "4", "8", "16"):
            rows = [r for r in rows if not (r["token"] == t and r["expert"] == expert)]
            rows.append(dict(base, token=t, topk=topk, block_m="16", ksplit="0", us1="0",
                             kernelName1=kernel, err1="0.0%", us2="0", kernelName2="",
                             err2="0.0%", us="0", run_1stage="1", xbf16="1", flat="1"))
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return f.name


def child(label: str) -> int:
    import torch

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us
    from suffix_hybrid.tools import moe_fp4_oracle as o  # builds the padded MXFP4 weights

    ok = True
    for M in MS:
        x = torch.randn(M, o.H, device=o.dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(o.E - 1, device=o.dev)[: o.TOPK - 1] for _ in range(M)])
        ids = torch.cat([ids, torch.full((M, 1), o.E - 1, device=o.dev)], 1).to(torch.int32)
        w = torch.cat([torch.softmax(torch.randn(M, o.TOPK - 1, device=o.dev), -1),
                       torch.sigmoid(torch.randn(M, 1, device=o.dev))], 1).float()
        try:
            ref, _ = o.run(x, w, ids, 128, tuned=False)
            out, kn = o.run(x, w, ids, 128, tuned=True)
            nan_rows = int(out.isnan().any(1).sum())
            cos = torch.nn.functional.cosine_similarity(
                out.nan_to_num().flatten(), ref.nan_to_num().flatten(), 0).item()
            good = nan_rows == 0 and cos >= 0.99
            os.environ["AITER_BYPASS_TUNE_CONFIG"] = "0"
            us = _graph_us(lambda i: o.fm.fused_moe(
                x, o.W1, o.W2, w, ids, None, o.ActivationType.Silu, o.QuantType.per_1x32, False,
                o.S1, o.S2, None, None, dtype=torch.bfloat16, hidden_pad=0,
                intermediate_pad=128, swiglu_limit=0.0), 8)[0]
            line = (f"{'PASS' if good else 'FAIL'} nan_rows={nan_rows} cos={cos:.5f} graphed "
                    f"{us:.1f} us {kn[0]}")
        except Exception as exc:  # noqa: BLE001 - a kernel that does not run is the datum
            good, line = False, f"ERROR {type(exc).__name__}: {str(exc)[:200]}"
        ok &= good
        print(f"{MARK} {label} M={M}: {line}", flush=True)
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        return child(sys.argv[2])
    env = dict(os.environ, SUFFIX_ROCM_AITER_FLYDSL_ZBUF="1")  # the shipped FlyDSL rows need it
    rc = 0
    for label, csv_path in [("shipped", SHIPPED)] + [(k, flat_csv(v)) for k, v in KERNELS.items()]:
        r = subprocess.run([sys.executable, "-m", "suffix_hybrid.tools.moe_flat_probe", "--child",
                            label], env=dict(env, AITER_CONFIG_FMOE=csv_path))
        print(f"{MARK} {label}: exit {r.returncode}", flush=True)
        rc |= r.returncode if label == "shipped" else 0  # FLAT failures are data, not a gate
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
