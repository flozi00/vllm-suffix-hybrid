# SPDX-License-Identifier: Apache-2.0
"""Activation-amax recorder (``SUFFIX_ACT_AMAX_RECORD=<globs>``, default off):
calibration for NVFP4 dense layers whose input bound cannot be proved from the
weights (Qwen4Exp hyper-connections, gated attention, GDN in/out projections).

Every allowlisted LinearBase still on UnquantizedLinearMethod after the
dense conversions gets a forward pre-hook folding max|x| into a 0-dim fp32
device buffer (in-place torch.maximum: no host sync). A daemon thread logs
``[suffix act-amax] rank=R {name: amax}`` every SUFFIX_ACT_AMAX_EVERY_S
(default 60) seconds; take the max over ranks and over the log and feed it,
with headroom, into SUFFIX_NVFP4_DENSE_ACT_AMAX. Run the calibration pod
with --enforce-eager (hooks under torch.compile/CUDA graphs are not relied on).
"""
import fnmatch
import json
import os
import threading

ENV = "SUFFIX_ACT_AMAX_RECORD"
MARK = "[suffix act-amax]"


def enabled() -> bool:
    return bool(os.environ.get(ENV, "").strip())


def globs() -> tuple:
    return tuple(g.strip() for g in os.environ.get(ENV, "").split(",") if g.strip())


def attach(model) -> int:
    import torch
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

    pats, bufs = globs(), {}
    for name, mod in model.named_modules():
        if not (isinstance(mod, LinearBase) and type(mod.quant_method) is UnquantizedLinearMethod):
            continue
        if not any(fnmatch.fnmatchcase(name, g) for g in pats):
            continue
        buf = torch.zeros((), dtype=torch.float32, device=mod.weight.device)

        def hook(_m, inp, _b=buf):
            torch.maximum(_b, inp[0].detach().abs().amax().float(), out=_b)

        mod.register_forward_pre_hook(hook)
        bufs[name] = buf
    if bufs:
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        every = float(os.environ.get("SUFFIX_ACT_AMAX_EVERY_S", "60"))

        def dump():
            ev = threading.Event()
            while not ev.wait(every):
                vals = {n: round(float(b.cpu()), 4) for n, b in bufs.items()}
                print(f"{MARK} rank={rank} {json.dumps(vals)}", flush=True)

        threading.Thread(target=dump, daemon=True, name="suffix-act-amax").start()
    print(f"{MARK} recording {len(bufs)} linears (globs={','.join(pats)})", flush=True)
    return len(bufs)
