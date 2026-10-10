# SPDX-License-Identifier: Apache-2.0
"""Boot-time JIT of vLLM's top-k/top-p sampler Triton kernels (V2 runner).

Why: vLLM 0.30's V1 ``TopKTopPSampler.__init__`` calls
``register_top_k_top_p_warmups()`` (v1/sample/ops/topk_topp_sampler.py), so
its JitWarmupRegistry compiles every variant of ``_topk_topp_kernel`` and
the split top-p pipeline (``_topp_sb_{stats,step,mask}_kernel``) at boot.
The V2 runner's ``Sampler`` (v1/worker/gpu/sample/sampler.py) calls
``apply_top_k_top_p`` directly and never registers them; V2 warmup_kernels
only samples at num_reqs=max_num_seqs, 2 and 1. Every other batch size picks
a new variant (split count S is a constexpr of batch size and SM count;
BATCH_SIZE / strides carry Triton's ==1 / %16 int specialization), so the
first real requests JIT mid-serving: the jit_monitor's "Triton kernel JIT
compilation during inference: _topp_sb_*" lines and a 3-7 s TTFT spike on
gemma4 (generation_config top_k 64 / top_p 0.95).

What: after the worker's V2 ``warmup_kernels`` (before CUDA-graph capture and
before the jit_monitor activates), compile those four dispatchers over vLLM's
OWN warmup domain (``get_warmup_keys(vllm_config=...)``: modes {k, k+p, p},
row strides {16, 2}, batch 1..max_num_seqs*(1+num_speculative_tokens), so
spec-decode verify rows are covered). Compilation goes through Triton's
``kernel.warmup`` with metadata-only inputs: no launch, no allocation, no
request/RNG state touched. min_p needs nothing: its kernel has no
batch-dependent specialization and V2 warmup_kernels already runs min_p.

Gate: SUFFIX_SAMPLER_WARMUP (default ON, =0 disables). Fail-soft: a failure
is a logged line, serving proceeds (first requests then JIT as before).
"""
import functools
import importlib.util
import os
import sys
import time

GATE = "SUFFIX_SAMPLER_WARMUP"
TARGET = "vllm.v1.worker.gpu_worker"
_MARK = "_suffix_sampler_warmup"


def _say(msg: str) -> None:
    print(f"[suffix sampler-warmup] {msg}", file=sys.stderr, flush=True)


def enabled() -> bool:
    return os.environ.get(GATE, "1").strip() != "0"


def _kernels():
    from vllm.platforms import current_platform
    from vllm.v1.sample.ops import topk_topp_triton as t
    # Same list/guard as vLLM's register_top_k_top_p_warmups().
    if current_platform.is_cpu():
        return []
    ks = [t._topk_topp]
    if current_platform.is_cuda_alike():
        ks += [t._topp_split_stats, t._topp_split_step, t._topp_split_mask]
    return ks


def warm(model_runner, kernels=None) -> None:
    """Compile every sampler variant serving can hit. Raises on failure."""
    if model_runner.is_pooling_model or not getattr(model_runner, "is_last_pp_rank", True):
        return
    vc = model_runner.vllm_config
    t0 = time.perf_counter()
    names, n = [], 0
    for k in _kernels() if kernels is None else kernels:
        keys = k.get_warmup_keys(vllm_config=vc)
        k.compile_many(keys)
        n += len(keys)
        names.append(getattr(getattr(k, "kernel", None), "__name__", None)
                     or getattr(k, "__name__", repr(k)))
    _say(f"warmed: {','.join(names)} ({n} variants) in "
         f"{(time.perf_counter() - t0) * 1e3:.0f} ms")


def _patch(mod) -> None:
    orig = mod.warmup_kernels
    if getattr(orig, _MARK, False):
        return

    @functools.wraps(orig)
    def warmup_kernels(model_runner, *a, **kw):
        out = orig(model_runner, *a, **kw)
        try:
            warm(model_runner)
        except Exception as exc:  # noqa: BLE001 - latency-only feature
            _say(f"FAILED (serving unaffected; first requests may JIT): {exc!r}")
        return out

    setattr(warmup_kernels, _MARK, True)
    mod.warmup_kernels = warmup_kernels


def install_post_import_hook() -> None:
    """Patch gpu_worker.warmup_kernels when that module loads. Never raises."""
    if not enabled():
        return
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
        return
    if any(getattr(f, _MARK, False) for f in sys.meta_path):
        return

    class _Finder:
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != TARGET:
                return None
            # Step aside first: importlib.util.find_spec walks sys.meta_path.
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real_exec = spec.loader.exec_module

            def exec_module(module, *a, **kw):
                real_exec(module, *a, **kw)
                try:
                    _patch(module)
                except Exception as exc:  # noqa: BLE001
                    _say(f"patch failed (warmup off): {exc!r}")

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    finder = _Finder()
    setattr(finder, _MARK, True)
    sys.meta_path.insert(0, finder)
