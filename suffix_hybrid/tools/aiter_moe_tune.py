# SPDX-License-Identifier: Apache-2.0
"""Tune AITER fused MoE on this card (boot gate aiter_moe_tune).

AITER ships tuned fused-MoE rows for 256-CU gfx950 only; on the 128-CU MI350P
every lookup falls back to heuristics (untuned FlyDSL 2-stage + separate sort
and quant launches), never to the one-stage asm FLAT kernel AMD's own 512-expert
top-10 rows pick for small token counts. This runs AITER's own tuner
(csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py, shipped for its JIT) for our
shape and prints the tuned CSV between markers; ship it in the bundle and point
AITER_CONFIG_FMOE at it.

    python -m suffix_hybrid.tools.aiter_moe_tune [--model-dim 2560 --inter-dim 768
        --expert 512 --topk 10 --tokens 1,2,4,8,16,32,64,128,256]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

MARK = "[suffix moe-tune]"
TUNER = "ck_gemm_moe_2stages_codegen/gemm_moe_tune.py"
HEADER = ("token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,"
          "q_type,use_g1u1,doweight_stage1")
ROW = ("{t},{d},{i},{e},{k},ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,"
       "torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0")


def say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def find_tuner() -> str | None:
    import aiter

    roots = {os.path.dirname(aiter.__file__), os.path.dirname(os.path.dirname(aiter.__file__))}
    roots |= {os.path.join(r, "aiter_meta") for r in list(roots)}
    for root in sorted(roots) + ["/app/aiter", "/opt/aiter"]:
        for path in glob.glob(os.path.join(root, "**", TUNER), recursive=True):
            return path
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", type=int, default=2560)
    ap.add_argument("--inter-dim", type=int, default=768)
    ap.add_argument("--expert", type=int, default=512)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tokens", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--timeout", type=int, default=2700)
    a = ap.parse_args()
    tuner = find_tuner()
    if tuner is None:
        say("tuner not found in the aiter install")
        return 0
    work = tempfile.mkdtemp(prefix="moe-tune-")
    untuned, tuned = os.path.join(work, "untuned.csv"), os.path.join(work, "tuned.csv")
    with open(untuned, "w") as fh:
        fh.write(HEADER + "\n")
        for t in a.tokens.split(","):
            fh.write(ROW.format(t=int(t), d=a.model_dim, i=a.inter_dim, e=a.expert, k=a.topk) + "\n")
    say(f"tuner {tuner}; shape d={a.model_dim} i={a.inter_dim} E={a.expert} top{a.topk} "
        f"tokens {a.tokens}")
    try:
        rc = subprocess.run([sys.executable, tuner, "--untune_file", untuned, "--tune_file", tuned],
                            cwd=os.path.dirname(tuner), timeout=a.timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    say(f"tuner exit {rc}")
    if os.path.exists(tuned):
        say("CSV BEGIN")
        with open(tuned) as fh:
            for line in fh:
                print(f"{MARK} | {line.rstrip()}", flush=True)
        say("CSV END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
