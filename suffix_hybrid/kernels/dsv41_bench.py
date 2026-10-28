# SPDX-License-Identifier: Apache-2.0
"""K3-on-silicon + K4 harness for the dsv41 fusion kernels (one GPU, one command each).

    python -m suffix_hybrid.helion_kernels.dsv41_hc      oracle           # K3 on the card
    python -m suffix_hybrid.helion_kernels.dsv41_hc      bench            # K4: split2 / resident vs the stock launches
    python -m suffix_hybrid.helion_kernels.dsv41_hc      capture_replay   # boot gates <kernel>_capture_replay
    python -m suffix_hybrid.helion_kernels.dsv41_router  bench --autotune # search a config first (slow)
    python -m suffix_hybrid.helion_kernels.dsv41_swiglu  oracle
    python -m suffix_hybrid.helion_kernels.dsv41_engram  oracle
    compute-sanitizer --tool memcheck python -m suffix_hybrid.helion_kernels.dsv41_hc capture_replay

Every family module's ``main`` calls ``main(module, argv)`` for oracle / per-kernel bench. It walks the
module's ``KERNELS`` (the lab ABI, so the numbers are the same ones the autotune lab sees):

  oracle  ours (Helion, a config per bucket) vs the entry's exact reference with the entry's own ``check``;
          exit status 1 on any FAIL. Where a kernel emits an MXFP8 activation, its bytes are also compared
          with vLLM's own quantiser (SILICON_CHECKS: flashinfer parity, the one thing CPU cannot prove).
  bench   ours vs ``baseline`` (the stock launches: vLLM TileLang mHC, cuBLAS / ll_bf16 + dsv4_topk,
          SiluAndMulWithClamp + flashinfer quantise, ...) from CUDA graphs with a COLD L2 (serving's regime:
          a layer's weights were last touched a whole step ago), median of N replays. Prints per bucket us,
          speedup, and the config row to ship ONLY where ours wins by > 2 % (AGENTS.md K4: a kernel ships
          only for the shapes where it wins).

Config source: --config FILE.json = {"<numtokens>": {helion.Config kwargs}}, else the module's
``default_config(case)`` heuristic (a starting point, NOT a tuned config), else --autotune
(helion.kernel without a config: LFBO search at the lab's effort). Silicon only; the CPU side of K3 is
tests/test_dsv41_*.py. SUFFIX_DSV41_FUSE_LAB_DEVICE=cpu runs ``oracle`` on CPU in Helion's interpret mode
(HELION_INTERPRET=1) as a smoke test of this harness.
"""
import argparse
import json
import statistics


def _cpu(a):
    import torch
    if isinstance(a, torch.Tensor):
        return a.detach().cpu()
    if isinstance(a, (tuple, list)):
        return type(a)(_cpu(x) for x in a)
    return a


def graph_us(fn, iters: int = 100, flush_mb: int = 512) -> float:
    """Median us of one CUDA-graph replay of fn(), L2 flushed before every replay."""
    import torch
    scratch = torch.empty(flush_mb << 20, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        scratch.add_(1)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def build(entry, module, case, cfg_file, autotune):
    """-> (compiled kernel, the config dict used or None when autotuning)."""
    import helion
    body = entry["body"]
    if autotune:
        return helion.kernel(body, static_shapes=False), None
    cfg = None
    if cfg_file:
        cfg = json.load(open(cfg_file)).get(str(case["numtokens"]))
    if cfg is None and hasattr(module, "default_config"):
        try:
            cfg = module.default_config(case, entry["name"])
        except TypeError:
            cfg = module.default_config(case)
    if cfg is None:
        raise SystemExit(f"no config for {case}: pass --config or --autotune")
    return helion.kernel(body, config=helion.Config(**cfg), static_shapes=False), cfg


def silicon_check(module, entry, args, got):
    """Optional on-silicon parity of an emitted MXFP8 activation with vLLM's own quantiser."""
    from suffix_hybrid.helion_kernels import dsv41_common as cm
    fn = getattr(module, "SILICON_CHECKS", {}).get(entry["name"])
    if fn is None:
        return True, ""
    if cm.DEVICE == "cpu":
        return True, "(cpu stand-in: flashinfer parity skipped)"
    return fn(args, got)


def main(module, argv=None) -> int:
    ap = argparse.ArgumentParser(prog=module.__name__)
    ap.add_argument("cmd", choices=["oracle", "bench"])
    ap.add_argument("--config")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--buckets", type=int, nargs="*")
    ap.add_argument("--model", default='{"architecture": "DeepseekV41ForCausalLM", "tp": 8, "hidden": 5120}')
    ap.add_argument("--iters", type=int, default=100)
    ns, _ = ap.parse_known_args(argv)
    import torch
    major = torch.cuda.get_device_capability()[0]
    if major != 12:
        raise SystemExit(f"{module.__name__}: sm_{major}x (need SM120)")
    model = json.loads(ns.model)
    status = 0
    for entry in module.KERNELS:
        cases = [c for c in entry["cases"](model) if not ns.buckets or c["numtokens"] in ns.buckets]
        print(f"== {entry['name']}  baseline: {entry['baseline_name']}")
        for case in cases:
            kern, cfg = build(entry, module, case, ns.config, ns.autotune)
            args = entry["make"](case, 0)
            ours = kern(*args)
            ref = entry["reference"](*args)
            ok, msg = entry["check"](ours, ref)
            ok2, msg2 = silicon_check(module, entry, args, ours)
            ok &= ok2
            line = f"  T={case['numtokens']:<4d} {'PASS' if ok else 'FAIL'}  {msg}{('  ' + msg2) if msg2 else ''}"
            if not ok:
                status = 1
            if ns.cmd == "bench" and ok:
                bargs = (entry.get("make_baseline") or entry["make"])(case, 0)
                t_ours = graph_us(lambda: kern(*args), ns.iters)
                t_base = graph_us(lambda: entry["baseline"](*bargs), ns.iters)
                sp = t_base / t_ours
                line += f"  ours {t_ours:7.1f} us  stock {t_base:7.1f} us  x{sp:.2f}"
                if sp > 1.02:
                    line += "   WIN -> ship " + json.dumps({"key": case, "config": cfg or "autotuned"})
            print(line, flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit("run a family module: python -m suffix_hybrid.helion_kernels.dsv41_hc bench")
