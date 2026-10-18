"""Fixed-config cold-cache diagnostic, before serving; never changes dispatch.

Recheck existing Pro configs and the quarantined 5090 candidate on the actual
card. No search or config writes. Timing follows inference-console's lab:
rotated input sets, CUDA graph replay, CUDA events, both orders. Each config
must first pass K2's independent dequantized reference.
"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics


def copies_for(touched, allocated, l2):
    return max(1, min(256, math.ceil(3 * l2 / touched), (6 << 30) // allocated))


def allocation_bytes(args):
    seen, total = set(), 0
    for t in args:
        if not hasattr(t, "untyped_storage"):
            continue
        s = t.untyped_storage()
        key = (str(t.device), s.data_ptr())
        if key not in seen:
            seen.add(key)
            total += s.nbytes()
    return total


def time_us(fn, sets):
    # Same graph/event protocol as api/autotune_runner.py::_time_us.
    import torch

    for args in sets:
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for args in sets:
            fn(*args)
    graph.replay()
    torch.cuda.synchronize()
    start, stop = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(20):
        graph.replay()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / 20 / len(sets)


def screen():
    import torch
    from suffix_hybrid.helion_kernels import k2_attn as K

    K.hk.ensure_vendor()
    import helion
    import triton

    helion_version = importlib.metadata.version("helion")

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    l2 = props.L2_cache_size
    assert l2 > 0 and torch.cuda.get_device_capability(device)[0] == 12
    table = K.hk.table(
        "k2_attn", "nvidia_rtx_pro_6000_blackwell_max_q_workstation_edition"
    )
    pro_choices = list(table.values())
    payload = json.loads(
        Path(__file__).with_name("gemma_k2_quarantined_config.json").read_text()
    )
    assert len(pro_choices) == 2
    results = []
    points = [
        (256, 8, 1, 4),
        (256, 8, 1, 2),
        (256, 8, 1, 1),
        (256, 8, 8, 1),
        (512, 2, 32, 4),
        (512, 2, 32, 6),
    ]
    for dim, kv_heads, batch, width in points:
        window = 1023 if dim == 256 else -1
        c = K.case(dim, 16, kv_heads, 64, window, batch, width)
        choices = (
            pro_choices + [payload["local"]["config"]]
            if dim == 256
            else [payload["full"]["config"]]
        )

        # Match the full-attention LFBO's 512..16k serving-context spread.
        def context(i, n):
            return K._ctx(i, n) if dim == 256 else int(512 * 32 ** (i / max(1, n - 1)))

        def make(seed):
            return K._problem(
                dim, 16, kv_heads, 64, window, batch, width, seed, "cuda", ctx=context
            )

        args = make(0)
        ref = K.reference(*args)
        touched = K.working_set_bytes(args)
        allocated = allocation_bytes(args)
        copies = copies_for(touched, allocated, l2)
        sets = [args] + [make(s) for s in range(1, copies)]

        def oxide(*a):
            return K.baseline(
                a[0],
                *K.natural_views(a),
                *a[5:9],
                width,
                window,
                a[14] / K.LOG2E,
                a[15],
            )

        for idx, config in enumerate(choices):
            rec = {
                "case": c,
                "actual_width": width,
                "config_id": idx,
                "config": config,
                "copies": copies,
                "touched_bytes": touched,
                "allocated_bytes": allocated,
                "l2_bytes": l2,
                "rotation_l2_ratio": touched * copies / l2,
                "max_copies": 256,
                "gpu": props.name,
                "torch": torch.__version__,
                "helion": helion_version,
                "triton": triton.__version__,
                "cuda_runtime": torch.version.cuda,
                "static_shapes": False,
                "source_sha256": hashlib.sha256(
                    Path(K.__file__).read_bytes()
                ).hexdigest(),
            }
            try:
                kernel = helion.kernel(
                    K._k2_nvfp4_attn, config=helion.Config(**config), **K.SETTINGS
                )
                q0 = args[0].clone()
                cache0 = [t.clone() for t in K.natural_views(args)]
                out = kernel(*args).clone()
                correct, check = K.check(out, ref)
                assert torch.equal(args[0], q0) and all(
                    torch.equal(a, b) for a, b in zip(K.natural_views(args), cache0)
                ), "input mutation"
                native_ok, native_check = K.check(oxide(*args).clone(), ref)
                rec.update(
                    correct=correct and native_ok,
                    check=check,
                    native_check=native_check,
                )
                del q0, cache0, out
                checks = [
                    {
                        "seed": 0,
                        "correct": rec["correct"],
                        "check": check,
                        "native_check": native_check,
                    }
                ]
                # Distinct data and page permutations, before accepting timing.
                # The seed-zero check alone cannot establish layout correctness.
                for seed in (13, 37):
                    probe = make(seed)
                    oracle = K.reference(*probe)
                    inputs = [probe[0], *K.natural_views(probe)]
                    saved = [t.clone() for t in inputs]
                    ok, msg = K.check(kernel(*probe).clone(), oracle)
                    native_ok, native_msg = K.check(oxide(*probe).clone(), oracle)
                    assert all(torch.equal(t, old) for t, old in zip(inputs, saved)), (
                        "input mutation"
                    )
                    checks.append(
                        {
                            "seed": seed,
                            "correct": ok and native_ok,
                            "check": msg,
                            "native_check": native_msg,
                        }
                    )
                    del probe, oracle, inputs, saved
                rec["oracle_checks"] = checks
                rec["correct"] = all(c["correct"] for c in checks)
                if rec["correct"]:
                    hs, ns = [], []
                    for first in [True, False]:
                        if first:
                            hs.append(time_us(kernel, sets))
                            ns.append(time_us(oxide, sets))
                        else:
                            ns.append(time_us(oxide, sets))
                            hs.append(time_us(kernel, sets))
                    rec.update(
                        helion_us=statistics.median(hs),
                        baseline_us=statistics.median(ns),
                        helion_samples_us=hs,
                        baseline_samples_us=ns,
                    )
                    rec["speedup"] = rec["baseline_us"] / rec["helion_us"]
            except Exception as exc:
                rec.update(correct=False, error=f"{type(exc).__name__}: {exc}"[:1000])
            results.append(rec)
            print("K2-COLD-RESULT " + json.dumps(rec, sort_keys=True), flush=True)
        del sets, args, ref
        torch.cuda.empty_cache()
    return 0 if results and all(r.get("correct") for r in results) else 1


def main():
    # An external parent can stop a CUDA sync or compiler that does not
    # return to Python signal handling. Kill only this diagnostic's group.
    import os
    import signal
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, __file__, "--worker"], start_new_session=True
    )
    try:
        return proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        print("K2-COLD-ERROR diagnostic exceeded 300 seconds", flush=True)
        return 1


if __name__ == "__main__":
    import sys

    raise SystemExit(screen() if sys.argv[1:] == ["--worker"] else main())
