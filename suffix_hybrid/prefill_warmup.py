# SPDX-License-Identifier: Apache-2.0
"""Prefill-shape boot warmup: fire the Triton JITs serving hits at real
prefill token counts BEFORE the first request (V2 runner). Gated by
SUFFIX_PREFILL_WARMUP=1 (default off); called by sampler_warmup's wrapper.

Why: vLLM's V2 warmup_kernels (vllm/v1/worker/gpu/warmup.py at 3f1bcfd)
prefills only decode_query_len+1 = 2-token prompts, so kernels whose compile
key depends on prefill M or sequence length first compile DURING inference.
vLLM's jit_monitor measured exactly that on lfm2.5-24b-a2b-fp8 @ RTX PRO 6000
(run 20261015T222306Z-rtx-pro-6000-9409 = trial 4871, serving window
22:28:33-22:29:36):
  _gumbel_sample_kernel    22:28:34  (temperature>0 with no top-k/top-p
                                      filter; for_sampler_warmup() and the
                                      greedy pass never take this path)
  _causal_conv1d_fwd_kernel 22:28:43 (long-sequence constexpr class; the
                                      stock warmup only ever ran 2-token
                                      sequences)
  fused_moe_kernel          22:29:02 (prefill-M tile config + Triton int
                                      specialization classes of
                                      num_valid_tokens/EM; the stock
                                      warmup's M never left small buckets)
The fused_moe JIT lands ~12 s into the c8 level, stalls the engine step and
the whole first wave of 8 requests -> c8 TTFT p90 1052.66 ms vs 204.72 ms
baseline (0e71 and its repeat 4871, learning f97a33).

What: after the worker's own warmup_kernels, replay vLLM's OWN
run_mixed_prefill_decode_warmup() (same module, normal scheduler inputs:
forged SchedulerOutputs, the worker's execute_model, block reservation via
the reservation policy, kv-connector disabled, cleanup) at a sweep of total
token counts up to the scheduler's max_num_batched_tokens, each point run at
N and N+1 so both of Triton's %16 int-specialization classes compile per
tile-config bucket; then one temperature>0 unfiltered prefill + sample so
the gumbel path compiles. Numerics are untouched: the same kernels with the
same configs compile, only EARLIER (bitwise-identical serving outputs).

Fail-soft everywhere: every stage is wrapped, a failure logs one
"[suffix prefill-warmup]" line and serving continues exactly as an
un-warmed server (first requests JIT, as today).

Env:
  SUFFIX_PREFILL_WARMUP=1        arm (default off)
  SUFFIX_PREFILL_WARMUP_SIZES    comma list of total scheduled tokens
                                 (default 512,1024,2048,3072,4096,6144,8192;
                                 every n also runs n+1; each value is capped
                                 at max(max_num_batched_tokens,
                                 max_model_len+1)-feasible sizes)
"""
import os
import sys
import time

GATE = "SUFFIX_PREFILL_WARMUP"
DEFAULT_SIZES = "512,1024,2048,3072,4096,6144,8192"


def enabled() -> bool:
    return os.environ.get(GATE, "").strip() == "1"


def _say(msg: str) -> None:
    print(f"[suffix prefill-warmup] {msg}", file=sys.stderr, flush=True)


def _cap(model_runner) -> int:
    """Largest total-num-tokens a serving prefill step can schedule."""
    mbt = getattr(getattr(model_runner, "scheduler_config", None),
                  "max_num_batched_tokens", 0) or 0
    mlen = getattr(model_runner, "max_model_len", 0) or 0
    caps = [x for x in (mbt, mlen + 1) if x > 0]
    return min(caps) if caps else 0


def _sizes(model_runner):
    """Token counts to replay: each base n and n+1 (both %16 spec classes),
    capped to what a real chunked-prefill step can schedule; plus the cap and
    cap-1 so the largest bucket's both classes always compile."""
    cap = _cap(model_runner)
    raw = os.environ.get("SUFFIX_PREFILL_WARMUP_SIZES", DEFAULT_SIZES)
    out = set()
    if cap >= 3:
        out.add(cap)
        if cap >= 4:
            out.add(cap - 1)
    for item in raw.split(","):
        item = item.strip()
        try:
            n = int(item)
        except ValueError:
            continue
        for v in (n, n + 1):
            if v >= 3 and (not cap or v <= cap):
                out.add(v)
    return sorted(out)


def _sweep(model_runner, worker_execute_model, worker_sample_tokens) -> None:
    """Replay vLLM's mixed prefill+decode warmup at serving token counts."""
    import vllm.v1.worker.gpu.warmup as vw

    mixed = getattr(vw, "run_mixed_prefill_decode_warmup", None)
    if mixed is None:
        _say("run_mixed_prefill_decode_warmup not found at this vLLM "
             "revision; sweep skipped")
        return
    sizes = _sizes(model_runner)
    if not sizes:
        _say("no feasible sizes; skipped")
        return
    t_all = time.perf_counter()
    for n in sizes:
        t0 = time.perf_counter()
        try:
            ok = mixed(model_runner, worker_execute_model,
                       worker_sample_tokens, n)
            _say(f"mixed warmup num_tokens={n}: "
                 f"{'ran' if ok else 'skipped (limits/blocks)'} "
                 f"in {(time.perf_counter() - t0) * 1e3:.0f} ms")
        except Exception as exc:  # noqa: BLE001 - fail-soft per point
            _say(f"mixed warmup num_tokens={n} FAILED (continuing): {exc!r}")
    _say(f"sweep done: {len(sizes)} sizes in "
         f"{(time.perf_counter() - t_all) * 1e3:.0f} ms")


def _gumbel_pass(model_runner, worker_execute_model,
                 worker_sample_tokens) -> None:
    """One temperature>0 unfiltered prefill + sample: compiles
    _gumbel_sample_kernel, the sampler path neither for_sampler_warmup()
    nor the greedy pass exercises (run 9409: JIT 1 s into serving)."""
    import vllm.v1.worker.gpu.warmup as vw
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request

    prompt_len = 128
    req_id = "_suffix_pw_sample_"
    block_count = vw._warmup_block_counter(model_runner)
    kv_cache_specs = [g.kv_cache_spec
                      for g in model_runner.kv_cache_config.kv_cache_groups]
    counts = [block_count(prompt_len, s) for s in kv_cache_specs]
    if model_runner.kv_cache_config.num_blocks <= sum(counts):
        _say("gumbel pass skipped: not enough KV blocks")
        return

    next_block_id = 1

    def _alloc(n):
        nonlocal next_block_id
        ids = list(range(next_block_id, next_block_id + n))
        next_block_id += n
        return ids

    token_ids = list(range(prompt_len))
    # Serving shape: temperature>0 with default top_k/top_p (no filter) ->
    # the gumbel sampling path. The multi-row prefill also exercises the
    # multi-token sampling shape.
    sampling_params = SamplingParams(max_tokens=2, temperature=0.1)

    out = SchedulerOutput.make_empty()
    out.num_spec_tokens_to_schedule = model_runner.num_speculative_steps
    out.scheduled_new_reqs = [
        vw.NewRequestData.from_request(
            Request(req_id, token_ids, sampling_params, None),
            block_ids=tuple(_alloc(n) for n in counts),
            prefill_token_ids=token_ids,
        )
    ]
    out.num_scheduled_tokens = {req_id: prompt_len}
    out.total_num_scheduled_tokens = prompt_len

    cleanup = SchedulerOutput.make_empty()
    cleanup.finished_req_ids = {req_id}

    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(out)
        worker_sample_tokens(None)
        worker_execute_model(cleanup)
    finally:
        model_runner.kv_connector.set_disabled(False)
    _say("gumbel sample pass done")


def maybe_run(model_runner, args) -> None:
    """Entry from sampler_warmup's warmup_kernels wrapper. Never raises."""
    try:
        if not enabled():
            return
        if model_runner.is_pooling_model or not getattr(
                model_runner, "is_last_pp_rank", True):
            return
        if len(args) < 2:
            _say("warmup_kernels got no worker callables; skipped")
            return
        worker_execute_model, worker_sample_tokens = args[0], args[1]
        t0 = time.perf_counter()
        _sweep(model_runner, worker_execute_model, worker_sample_tokens)
        _gumbel_pass(model_runner, worker_execute_model, worker_sample_tokens)
        _say(f"prefill-shape warmup complete in "
             f"{(time.perf_counter() - t0) * 1e3:.0f} ms")
    except Exception as exc:  # noqa: BLE001 - never break boot/serving
        try:
            _say(f"FAILED (serving unaffected; first requests may JIT): "
                 f"{exc!r}")
        except Exception:
            pass
