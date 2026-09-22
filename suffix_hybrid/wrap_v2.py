# SPDX-License-Identifier: Apache-2.0
"""Opt-in, source-pinned V2 standard-MTP dense-row wire-up.

Native propose ALWAYS runs first on every TP rank. d05da62e9 model_runner
sample_tokens postprocesses GPU history before calling propose (~2124), then
persists our full-width output into req_states.draft_tokens (~2151) and hands
it to the scheduler via DraftTokensHandler (~2173). The next AR prefill
recomputes verified positions from target history/hidden states; no native
forward or KV repair is skipped. Never stitch a stale native tail onto suffix.

Synchronous owner-only D2H history transfer is intentional prototype overhead,
not a speed claim. No vocabulary-sized probabilities are allocated. CUDA/TP
correctness and throughput remain live benchmark gates.

Temperature gate (d05da62e9 rejection_sampler_utils): with
draft_sample_method="greedy" (draft_logits=None) the sampler treats the draft
as a point mass AT the proposed token, so replacing rows is statistically
consistent at any temperature. With draft_sample_method="probabilistic"
(draft_logits is a live [max_num_reqs, K, V] buffer the sampler reads for the
accept ratio) a replaced row at temperature>0 would be graded against the
NATIVE drafter's distribution — wrong statistics. In that mode we arbitrate
ONLY the temperature==0 rows (the sampler skips draft_logits entirely when
temp==0) and pass stochastic rows through native.

Install shape: install_v2() patches the V2 GPUModelRunner.load_model BEFORE
the engine builds it; after init_speculator resolved the concrete method
class, the speculator INSTANCE gets its propose wrapped once. Every per-step
property that cannot be arbitrated (dummy/profile/capture calls, shape drift,
any mixer raise) DEGRADES to the untouched native draft with a rate-limited
stderr line — raising inside propose propagates through sample_tokens as an
EngineCore fatal and kills the pod.
"""
import functools
import inspect
import os
import sys

import torch


# Speculator classes the wrapper may arbitrate: the native propose runs in
# full on every step, so the drafter's own KV/hidden chain stays
# self-consistent; we only rewrite the RETURNED draft rows. Enumerate every
# vendor spelling (classes were renamed across checkouts); an unknown
# speculator class degrades to native-only, never crashes.
_ACCEPT_SPECULATORS = frozenset({
    "MTPSpeculator",
    "EagleSpeculator",
    "DFlashSpeculator",
    "DFlash2Speculator",
    "DSparkSpeculator",
    "MultiModuleMTPSpeculator",
    "Gemma4Speculator",
})

# The propose() parameters the wrapper reads. A rename is a contract break:
# install fails closed (before the pod serves); per-step drift degrades.
_REQUIRED_PROPOSE_PARAMS = (
    "input_batch", "attn_metadata", "slot_mappings", "last_hidden_states",
    "aux_hidden_states", "num_sampled", "num_rejected", "last_sampled",
    "next_prefill_tokens", "temperature", "seeds",
)


def _check_speculator_capability(spec_cls):
    """Fail closed at install time when the live speculator lacks the shape.

    Pure introspection (no imports, no I/O) so unit tests can pass fakes.
    """
    propose = getattr(spec_cls, "propose", None)
    if not callable(propose):
        raise RuntimeError(
            f"suffix hybrid V2 unsupported speculator: {spec_cls.__name__} "
            "has no callable propose")
    try:
        params = set(inspect.signature(propose).parameters)
    except (TypeError, ValueError):
        return propose
    missing = [p for p in _REQUIRED_PROPOSE_PARAMS if p not in params]
    if missing:
        raise RuntimeError(
            f"suffix hybrid V2 unsupported speculator: "
            f"{spec_cls.__name__}.propose missing params {missing}")
    return propose


def _wrap_propose(runner, original, mixer, group, probabilistic=False):
    previous_widths = {}
    state = {"skips": 0, "reason": ""}

    def _degrade_line(kind):
        state["skips"] += 1
        if state["skips"] <= 3 or state["skips"] % 100 == 0:
            print(f"suffix_hybrid v2 native passthrough ({kind}, "
                  f"#{state['skips']}): {state['reason']}",
                  file=sys.stderr, flush=True)

    @functools.wraps(original)
    def propose(input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False):
        native = original(input_batch, attn_metadata, slot_mappings,
                          last_hidden_states, aux_hidden_states, num_sampled,
                          num_rejected, last_sampled, next_prefill_tokens,
                          temperature, seeds, dp_sync=dp_sync,
                          dummy_run=dummy_run,
                          skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          mm_inputs=mm_inputs, is_profile=is_profile)
        # CUDA-graph capture / warmup / profile calls run the native forward
        # untouched: no CPU sync, no collectives, no state mutation inside a
        # captured region.
        if dummy_run or is_profile or skip_attn_for_dummy_run:
            return native
        try:
            output = native.clone()
            if group.rank_in_group == 0:
                idx = input_batch.idx_mapping.long()
                counts = runner.req_states.total_len.gpu[idx].detach().cpu()
                width = int(counts.max()) if counts.numel() else 0
                if width <= 0:
                    return group.broadcast(output, src=0)
                # Blocking .cpu() completes the postprocess producer stream
                # before numpy reads; never use optimistic CPU mirrors.
                history = (runner.req_states.all_token_ids.gpu[:, :width][idx]
                           .detach().cpu())
                ids = list(input_batch.req_ids)
                sampled = num_sampled.detach().cpu().tolist()
                rejected = num_rejected.detach().cpu().tolist()
                # Probabilistic rejection sampling grades each draft token
                # against the NATIVE drafter's stored logits; a suffix token
                # was not sampled from that q, so replacing a stochastic row
                # breaks the accept-ratio statistics. The sampler ignores
                # draft_logits entirely at temp==0 (greedy accept/resample),
                # so arbitrate only the temperature==0 rows in that mode.
                # Greedy mode (draft_logits=None) treats the draft as a point
                # mass at the proposed token — replacement is consistent at
                # any temperature, no per-row gate needed.
                greedy_rows = None
                if probabilistic:
                    greedy_rows = [t == 0.0 for t in
                                   temperature[idx].detach().cpu().tolist()]
                accepted = []
                for req_id, ns, nr in zip(ids, sampled, rejected):
                    a, verified = ns - 1, ns + nr - 1
                    previous = previous_widths.get(req_id, 0)
                    valid = (ns > 0 and nr >= 0 and 0 < verified <= previous
                             and 0 <= a <= verified
                             and (a < verified or verified == previous))
                    accepted.append(a if valid else -1)
                mixed = mixer.mix_numpy(
                    ids, counts.numpy(), history.numpy(),
                    native.detach().cpu().tolist(), accepted)
                # Fixed-width row write: the runner persists
                # [num_reqs, K] into req_states.draft_tokens, so every row
                # keeps K slots. Slots past the mixed proposal retain the
                # NATIVE draft token (a valid draft, just not suffix-
                # extended) — never -1: the scheduler treats each entry as a
                # schedulable spec token, and a shortened row would change
                # verification width semantics this build does not support.
                # A mixed row LONGER than K is a mixer bug: degrade.
                k = output.shape[1]
                for i, row in enumerate(mixed):
                    if len(row) > k:
                        raise ValueError(
                            f"mixed row {i} length {len(row)} exceeds K={k}")
                    if row and (greedy_rows is None or greedy_rows[i]):
                        output[i, :len(row)] = torch.tensor(
                            row, dtype=output.dtype, device=output.device)
                previous_widths.clear()
                previous_widths.update(zip(ids, map(len, mixed)))
            return group.broadcast(output, src=0)
        except Exception as exc:
            state["reason"] = f"{type(exc).__name__}: {exc}"
            _degrade_line("degraded")
            return native
    return propose


def install_v2():
    """Patch the V2 runner's speculator path. Returns True when installed.

    Returns False (never raises) when the V2 runner module does not exist in
    this vLLM — the caller then falls back to the V1 hook. Raises RuntimeError
    only when V2 IS present but its importable contracts are broken, so an
    enabled-but-incompatible worker is visible in the pod log, never silent.
    """
    if os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() != "1":
        return False
    try:
        module = __import__(
            "vllm.v1.worker.gpu.model_runner", fromlist=["GPUModelRunner"])
    except ImportError:
        return False
    try:
        from suffix_hybrid._native import HybridMixer
        from vllm.distributed.parallel_state import get_tp_group
        spec_mod = __import__(
            "vllm.v1.worker.gpu.spec_decode.speculator",
            fromlist=["DraftModelSpeculator"])
        DraftModelSpeculator = spec_mod.DraftModelSpeculator
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "suffix hybrid V2 installation requires vLLM V2 runner and the "
            "Rust HybridMixer") from exc

    original_load = module.GPUModelRunner.load_model
    if getattr(original_load, "_suffix_hybrid_v2", False):
        return True

    @functools.wraps(original_load)
    def load_model(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        try:
            speculator = getattr(self, "speculator", None)
            if speculator is None or not isinstance(
                    speculator, DraftModelSpeculator):
                print("suffix_hybrid v2: no DraftModelSpeculator after "
                      "load_model — hook inactive (plain native decode).",
                      file=sys.stderr, flush=True)
                return result
            cls_name = type(speculator).__name__
            if cls_name not in _ACCEPT_SPECULATORS:
                print(f"suffix_hybrid v2 WARNING: unaccepted speculator "
                      f"class {cls_name} — staying native-only.",
                      file=sys.stderr, flush=True)
                return result
            _check_speculator_capability(type(speculator))
            k = int(getattr(speculator, "num_speculative_steps", 0) or 0)
            if k <= 0:
                print("suffix_hybrid v2: speculator has no draft steps — "
                      "hook inactive.", file=sys.stderr, flush=True)
                return result
            mixer = HybridMixer(
                k, int(getattr(speculator, "max_model_len", 0) or 0))
            group = get_tp_group()
            if getattr(speculator.propose, "_suffix_hybrid_hook", False):
                return result
            # draft_logits is allocated only for draft_sample_method=
            # "probabilistic" (d05da62e9 speculator.__init__): its presence
            # means the rejection sampler will grade every draft token
            # against the native drafter's stored distribution, so the
            # wrapper must pass stochastic rows through untouched.
            probabilistic = getattr(speculator, "draft_logits", None) \
                is not None
            # Instance-attribute bind: the runner calls
            # self.speculator.propose(input_batch=..., ...) with keywords, so
            # the wrapper closes over the speculator's BOUND class function
            # (self included) plus this runner/mixer/tp-group.
            bound = type(speculator).propose.__get__(speculator,
                                                     type(speculator))
            wrapped = _wrap_propose(self, bound, mixer, group,
                                    probabilistic=probabilistic)
            wrapped._suffix_hybrid_hook = True
            speculator.propose = wrapped
            print(f"suffix_hybrid v2 installed speculator={cls_name} "
                  f"k={k} tp={group.world_size} "
                  f"draft_sample_method="
                  f"{'probabilistic' if probabilistic else 'greedy'}",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            # load_model failure must not crash the worker: the hook is an
            # accelerator, not a correctness component.
            print(f"suffix_hybrid v2 install failed on this worker "
                  f"(native-only): {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        return result

    load_model._suffix_hybrid_v2 = True
    module.GPUModelRunner.load_model = load_model
    return True
