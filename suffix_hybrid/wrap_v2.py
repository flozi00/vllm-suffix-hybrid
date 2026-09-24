# SPDX-License-Identifier: Apache-2.0
"""Opt-in, source-pinned V2 standard-MTP dense-row wire-up.

Native propose ALWAYS runs first on every TP rank. d05da62e9 model_runner
sample_tokens postprocesses GPU history before calling propose (~2124), then
persists our full-width output into req_states.draft_tokens (~2151) and hands
it to the scheduler via DraftTokensHandler (~2173). The next AR prefill
recomputes verified positions from target history/hidden states; no native
forward or KV repair is skipped. Never stitch a stale native tail onto suffix.

Async pinned history mirror (replaces the old synchronous owner-only D2H of
the FULL history buffer every step): the wrapper keeps a CPU-authoritative
per-request token mirror and, once per real step, enqueues on a side CUDA
stream a BOUNDED optimistic window copy — for row i only
``all_token_ids[j, L : L+K+2]`` (L = current mirror length), plus
``total_len[j]``, ``num_sampled`` and ``num_rejected`` — into preallocated
pinned int64 staging, then records a CUDA event. The host never blocks in
the enqueue path; the copies are ordered behind an event recorded on the
producer stream so they cannot read half-postprocessed rows. The NEXT step,
before reading the mirror, we ``event.synchronize()`` — one full engine step
has passed, so the wait is nearly free — and absorb the tail. In steady
state the mirror lags the GPU by at most one step, which is acceptable
because every draft (suffix or native) is fully target-verified; the first
real step for a request may therefore legitimately pass through native. A
request is seeded once (first real step it appears: one scalar ``total_len``
read + one full-row copy, per request lifetime, never per step) so the
bounded window is always positioned at the live end of history; a window
overrun or a row reset re-seeds from position 0. This removes both mid-step
host syncs, the 16-100 MB/step full-buffer D2H, and the per-step per-row
tensor allocations that halved TP=8 throughput when the hook was live.

Ingestion rule: ``mix_numpy`` IS the corpus ingestion path (mix_core
inserts the observed history on every call), so NO cache-size gate may run
before it — gating on ``cache_tokens() == 0`` before the mix keeps the
cache cold forever (nothing ingests → tokens never leave 0; live-pod
regression, 2026-09-22). The cold mix provably echoes native, and the
unchanged-publish path below returns native untouched.
Write-back skip: when the written rows equal the native rows the wrapper
likewise publishes native untouched (counted as skips_unchanged).
Skipping the TP broadcast itself is only collective-safe
when every rank makes the same decision from rank-invariant inputs: that
holds for a single-rank group (no collective exists) and in replay mode
(every rank mirrors + mixes locally), so those paths return the native
object with no broadcast at all. In multi-rank broadcast mode the non-owner
mixer cache is never fed (only rank 0 mixes), so rank 0's unchanged
predicate is NOT visible to the slaves; there the skip still publishes via
``group.broadcast(native, src=0)`` — no clone, no D2H, but the
slaves' blocking receive must never be orphaned or the engine deadlocks.

Temperature gate (d05da62e9 rejection_sampler_utils): with
draft_sample_method="greedy" (draft_logits=None) the sampler treats the draft
as a point mass AT the proposed token, so replacing rows is statistically
consistent at any temperature. With draft_sample_method="probabilistic"
(draft_logits is a live [max_num_reqs, K, V] buffer the sampler reads for the
accept ratio) a replaced row at temperature>0 would be graded against the
NATIVE drafter's distribution — wrong statistics. In that mode we arbitrate
ONLY the temperature==0 rows (the sampler skips draft_logits entirely when
temp==0) and pass stochastic rows through native.

Env switches: SUFFIX_HYBRID_SYNC_HOOK=1 (checked every step) selects the old
fully-synchronous body verbatim for a live old-vs-new A/B.
SUFFIX_HYBRID_TP_MODE=replay (default broadcast) makes every rank run the
mirror+mix locally and publish with NO collective; replay assumes
rank-uniform native drafts (the drafter forward is collective over TP) and
is measurement-only — broadcast stays authoritative. If the async fast path
raises once it is permanently disabled for this wrapper and every later step
runs the synchronous body (logged once).

- Install shape: install_v2() patches the V2 GPUModelRunner.load_model BEFORE
  the engine builds it; after init_speculator resolved the concrete method
  class, the speculator INSTANCE gets its propose wrapped once. Every per-step
  property that cannot be arbitrated (dummy/profile/capture calls, shape drift,
  any mixer raise) DEGRADES to the untouched native draft with a rate-limited
  stderr line — raising inside propose propagates through sample_tokens as an
  EngineCore fatal and kills the pod.

SUFFIX-ONLY MODE (SUFFIX_HYBRID_SUFFIX_ONLY=1, gemma lane wake #6-7): the
GPU-drafter forward is ECONOMICALLY DEAD on Gemma-4-26B-A4B @ SM120 (measured:
k=4 spec-step = 10x native step cost, ceiling 5 tok/step -> max 0.50x even at
99.95% acceptance; k=1 3.27x vs 2x -> 0.61x). The only surviving golden-rule
path removes the drafter from the equation: the wrapper REPLACES propose()
outright — no draft forward, no drafter KV work — and fills draft_tokens
straight from the host-RAM suffix cache (suffix doctrine: zero VRAM).

Contract honored (vLLM v0.30.0, sources pinned in
bench/gemma-research/RESEARCH-vllm-v030-spec-contract.md):
- propose returns the persistent [num_reqs, K] int64 GPU buffer
  (speculator.draft_tokens); the runner scatters it into
  req_states.draft_tokens (model_runner.py:2158) — full width mandatory.
- draft_sample_method must be greedy (draft_logits stays None; the rejection
  sampler at model_runner.py:1563 needs drafter logits otherwise).
- dummy_run/is_profile calls must also return the correctly shaped buffer
  (zeros are fine) — the captured target graphs never contained the drafter.
- observe_verification stays native (no-op without adaptive verification).
Ragged per-row widths (miss row -> plain 1-token decode at 1x native cost):
the scheduler consumes request.spec_token_ids of ANY length per row
(scheduler.py:812-827 trims, update_draft_token_ids:2404+ accepts ragged
DraftTokenIds, prepare_inputs builds per-row cu_num_logits, rejection sampler
kernels are per-row via cu_num_draft_tokens). The stock
DraftTokensHandler.get_draft_tokens (spec_decode/utils.py:60-70) only D2Hs
real rows under structured output; otherwise it publishes K-wide [-1]
placeholders. The wrapper therefore ALSO patches get_draft_tokens to return
our true per-row widths (tracked on the host alongside each proposal) —
sync scheduling (async_scheduling defaults off; we do not pass it) calls it
every step via EngineCore.post_step -> update_draft_token_ids.
Rust-first (language contract): ALL per-step algorithm work lives in the
Rust V2SuffixProposer (src/mixer.rs) — per-request full-history mirrors,
64-token head+tail boundary continuity, cache lookups, ingestion on
departure, true-width tracking. The adapter below is pure contract wiring:
read totals (one blocking [n] D2H that also fences the UVA buffer writes
postprocess_sampled issued), pass the ZERO-COPY host view of
req_states.all_token_ids (UVA pinned memory, int32) plus ids/indices, and
upload the packed [n,K] result. postprocess_sampled runs BEFORE propose,
so the buffer read is the CURRENT authoritative row — no mirror lag, no
shift correction; the cache continuation starts at the true next token.
A miss simply proposes width 0 (ragged row -> 1x decode): correctness is
never at risk because every draft is target-verified.
"""
import contextlib
import functools
import inspect
import json
import os
import sys
import time

import numpy as np
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


def _sync_body(runner, mixer, group, probabilistic, previous_widths,
               state, interval):
    """Legacy per-step body, verbatim from the pre-mirror wire-up (A/B ref).

    Blocking owner-only D2H of the full history buffer every step, per-row
    tensor allocs, unconditional broadcast. Kept exactly as it shipped so
    SUFFIX_HYBRID_SYNC_HOOK=1 (and the fast path's sticky fallback) is a
    true old-vs-new comparison. Returns run(native, input_batch,
    num_sampled, num_rejected, temperature) -> published draft.
    """
    def run(native, input_batch, num_sampled, num_rejected, temperature):
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
            # Fixed-width row write: the runner persists [num_reqs, K] into
            # req_states.draft_tokens, so every row keeps K slots. Slots
            # past the mixed proposal retain the NATIVE draft token (a
            # valid draft, just not suffix-extended) — never -1: the
            # scheduler treats each entry as a schedulable spec token, and
            # a shortened row would change verification width semantics
            # this build does not support. A mixed row LONGER than K is a
            # mixer bug: degrade.
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
            state["mixes"] += 1
            if interval and state["mixes"] % interval == 0:
                print("suffix_hybrid_native " + json.dumps(
                    mixer.get_stats(), sort_keys=True),
                    file=sys.stderr, flush=True)
        return group.broadcast(output, src=0)
    return run


def _suffix_only_wrap(runner, speculator, mixer, group, k):
    """Replace propose() with drafter-free, Rust-owned suffix drafting.

    Language contract: NO per-step Python algorithm work. The adapter only
    (a) reads the UVA host-resident token buffer + total lengths (one tiny
    GPU event sync for total_len, then zero-copy CPU reads), (b) calls the
    Rust V2SuffixProposer once, (c) uploads the packed [n,k] drafts. All
    mirror bookkeeping, cache lookups, ingestion, and width tracking live
    in Rust; this function is pure vLLM-contract wiring.

    vLLM contract anchors (v0.30.0): postprocess_sampled runs BEFORE
    propose and writes the last sampled tokens into req_states.all_token_ids
    (UVA host buffer) + total_len (GPU StagedWriteTensor), so a read here
    sees the CURRENT authoritative row. Sync-scheduler ragged widths go out
    via the get_draft_tokens patch reading proposer.widths_table.
    """
    from suffix_hybrid._native import V2SuffixProposer
    state = {"skips": 0, "reason": ""}
    interval = int(os.environ.get("SUFFIX_HYBRID_LOG_INTERVAL", "0") or 0)
    min_len = max(int(os.environ.get(
        "SUFFIX_HYBRID_SUFFIX_MIN", "1") or 1), 1)
    uniform_k = os.environ.get("SUFFIX_HYBRID_UNIFORM_K", "").strip() == "1"
    proposer = V2SuffixProposer(
        int(k), int(getattr(speculator, "max_model_len", 0) or 0) or 32768,
        min_len, uniform_k)
    if uniform_k:
        print(f"suffix_hybrid v2 suffix-only UNIFORM-K pallet k={k} "
              f"(all rows width k, CUDA-graph matched)", file=sys.stderr,
              flush=True)
    draft_tokens = speculator.draft_tokens          # [max_num_reqs, K] GPU
    req_states = runner.req_states
    totals_gpu = req_states.total_len.gpu
    # ZERO-COPY host view of the full [max_num_reqs, max_model_len] token
    # buffer (UVA pinned memory, int32). Rust reads rows straight from it;
    # the only GPU traffic per step is the [n] totals D2H that also fences
    # the UVA writes (postprocess_sampled's post_update kernel completed).
    ats_np = req_states.all_token_ids._uva_buf.np
    lut = req_states.req_id_to_index
    max_reqs = int(getattr(speculator, "max_num_reqs", 32) or 32)

    # Per-step draft accounting trace (gemma wake #9): the engine rejected
    # ~96% of published drafts at position 1 even on verbatim-repeat
    # traffic, so either the drafted continuation is wrong (collisions) or
    # the buffer/totals read is misaligned by one step. TRACE compares, for
    # each traced request, the draft published LAST step against what the
    # engine actually wrote into the token row at those positions this
    # step — the raw diff is diagnostic either way.
    trace_cfg = {
        "on": os.environ.get("SUFFIX_HYBRID_TRACE", "").strip() != "",
        "budget": int(os.environ.get(
            "SUFFIX_HYBRID_TRACE_MAX", "200") or 200),
        "fired": False,
    }
    trace_state = {}   # rid -> (total_at_publish, [draft tokens])

    # One pinned [max_reqs] totals buffer, reused every step.
    pin = torch.cuda.is_available()
    totals_cpu = torch.zeros(
        (max_reqs,), dtype=torch.int64, pin_memory=pin)

    def _totals_now(n):
        # Blocking D2H of the first n totals; also acts as the fence for
        # the UVA buffer writes (same stream ordering as post_update).
        dst = totals_cpu[:n]
        dst.copy_(totals_gpu[:n], non_blocking=False)
        return dst.numpy()

    @functools.wraps(type(speculator).propose)
    def propose(input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False):
        n = int(input_batch.num_reqs)
        out = draft_tokens[:n]
        if dummy_run or is_profile:
            out.zero_()
            return out
        try:
            ids = list(input_batch.req_ids)
            totals = _totals_now(n)
            idx_np = np.fromiter((lut[rid] for rid in ids),
                                  dtype=np.int64, count=n)
            # Draft accounting: compare last step's published drafts against
            # the engine's actual writes at those positions this step. The
            # engine writes the verified/sampled tokens at [p_total .. ]
            # AFTER our propose returns, BEFORE the next propose — the UVA
            # row read here is last step's CLEARED row audio.
            if trace_cfg["on"]:
                for rid in list(trace_state):
                    if rid not in ids:
                        del trace_state[rid]   # request left the batch
                for i, rid in enumerate(ids):
                    prev = trace_state.get(rid)
                    if prev is None:
                        continue
                    p_total, p_drafts = prev
                    t_now = int(totals[i])
                    row_i = int(idx_np[i])
                    accepted = []
                    for j, dt in enumerate(p_drafts):
                        pos = p_total + j
                        real = int(ats_np[row_i, pos]) if pos < ats_np.shape[1] else -2
                        accepted.append((dt, real))   # pairing BEFORE replace
                    match = sum(1 for dt, re in accepted if dt == re)
                    print(f"suffix_hybrid TRACE {rid} total {p_total}->{t_now} "
                          f"width {len(p_drafts)} matched {match}/{len(p_drafts)} "
                          f"pairs {accepted[:6]}", file=sys.stderr, flush=True)
                    trace_cfg["budget"] -= 1
                    if trace_cfg["budget"] <= 0:
                        trace_cfg["on"] = False   # fire once, then be quiet
            packed, widths_np = proposer.propose_suffix_only(
                ids, idx_np, totals, ats_np)
            # Store drafts JUST published for next step's comparison.
            if trace_cfg["on"]:
                packed_l = packed.tolist()
                for i, rid in enumerate(ids):
                    w = int(widths_np[i])
                    if w > 0:
                        trace_state[rid] = (int(totals[i]), packed_l[i][:w])
                    elif rid in trace_state:
                        del trace_state[rid]
            out.zero_()
            out.copy_(torch.from_numpy(packed).to(
                out.device, out.dtype), non_blocking=pin)
            if interval and int(proposer.get_stats()["steps"]) % interval == 0:
                print("suffix_hybrid suffix-only " + json.dumps(
                    proposer.get_stats(), sort_keys=True, default=str),
                    file=sys.stderr, flush=True)
            return out
        except Exception as exc:
            state["reason"] = f"{type(exc).__name__}: {exc}"
            state["skips"] += 1
            if state["skips"] <= 3 or state["skips"] % 100 == 0:
                print(f"suffix_hybrid suffix-only passthrough "
                      f"(#{state['skips']}): {state['reason']}",
                      file=sys.stderr, flush=True)
            # Widths published by run() are now STALE: the packed upload
            # failed and out is zeroed. Retract them or the scheduler
            # verifies zeroed drafts at phantom widths.
            try:
                proposer.clear_widths()
            except Exception:
                pass
            out.zero_()
            return out
    propose._suffix_proposer = proposer
    # Token-level verify trace hook points (SUFFIX_HYBRID_TRACE_VERIFY2):
    # expose the per-rid published-drafts registry to the sampler wrap.
    setattr(propose, "_suffix_trace_state", trace_state)
    setattr(propose, "_suffix_trace_cfg", trace_cfg)
    return propose


def _wrap_rejection_sampler(runner, propose_hook, k):
    """Token-level verify trace (SUFFIX_HYBRID_TRACE_VERIFY2).

    Publishes, per traced step and rid:
      pub=  the drafts our propose published for that rid (from trace_state)
      graded= the token ids the rejection kernel will compare against target
            argmax — draft_sampled = input_ids[logits_indices] sliced by
            cu_num_logits per request, first row (last sampled token) dropped
      acc=  the post-verify sampled_token_ids row (accepted drafts up to the
            first rejection, then target argmax) and num_sampled delta
    Discriminates: handoff corruption (pub != graded -> scheduler/scatter
    bug) vs genuine argmax mismatch (pub == graded but acc rejects).

    NOTE(2026-09-24): special-method lookup ignores instance attributes, so
    the wrap binds on the CLASS (one sampler instance per process here).
    """
    sampler = getattr(runner, "rejection_sampler", None)
    if sampler is None:
        return False
    on_env = os.environ.get("SUFFIX_HYBRID_TRACE_VERIFY2", "").strip()
    if not on_env:
        return False
    budget = [int(os.environ.get("SUFFIX_HYBRID_TRACE_VERIFY2_MAX",
                                  "24") or 24)]
    original = type(sampler).__call__

    trace_state = getattr(propose_hook, "_suffix_trace_state", None) or {}

    @functools.wraps(original)
    def call(self, logits, input_batch, draft_logits=None, *args, **kwargs):
        out = original(self, logits, input_batch, draft_logits,
                       *args, **kwargs)
        if budget[0] <= 0 or not trace_state:
            return out
        try:
            if getattr(input_batch, "num_draft_tokens", 0) <= 0:
                return out
            cu = input_batch.cu_num_logits_np.tolist()
            graded = input_batch.input_ids[
                input_batch.logits_indices].detach().to(
                "cpu", non_blocking=False).tolist()
            sampled = out.sampled_token_ids.detach().to(
                "cpu", non_blocking=False).tolist()
            nsamp = out.num_sampled.detach().to(
                "cpu", non_blocking=False).tolist()
            rids = list(input_batch.req_ids)
            for i, rid in enumerate(rids):
                pub = trace_state.get(rid)
                if pub is None:
                    continue
                p_total, p_drafts = pub
                start, end = cu[i], cu[i + 1]
                w = end - start - 1
                if w <= 0:
                    continue
                g = graded[start + 1:end]
                p = p_drafts[:w]
                m = sum(1 for a, b in zip(p, g) if a == b)
                a_row = sampled[i][:w] if i < len(sampled) else []
                n_acc = max(int(nsamp[i]) - 1, 0) if i < len(nsamp) else -1
                print(f"suffix_hybrid VTRACE2 rid={rid!r} w={w} "
                      f"pubmatch={m}/{w} graded={g[:8]} pub={p[:8]} "
                      f"accrow={a_row[:8]} nacc={n_acc}",
                      file=sys.stderr, flush=True)
            budget[0] -= 1
            if budget[0] <= 0:
                print("suffix_hybrid VTRACE2 budget exhausted",
                      file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"suffix_hybrid VTRACE2 error: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr, flush=True)
            budget[0] = 0
        return out
    # Special-method lookup: obj(...) resolves __call__ on the TYPE, so the
    # wrap must bind class-level (one sampler instance per process here).
    type(sampler).__call__ = call
    return True


def _patch_get_draft_tokens(runner, widths_table):
    """Make DraftTokensHandler.get_draft_tokens return our ragged widths.

    The stock handler (v0.30.0 spec_decode/utils.py) publishes K-wide [-1]
    placeholder rows unless structured output forced a D2H. The scheduler
    happily consumes ragged rows (that is exactly how V1 ngram/suffix
    proposers feed it): only the LENGTH of each CPU row matters for
    scheduling — the verification tokens themselves come from the GPU
    req_states.draft_tokens via combine_sampled_and_draft_tokens. We trim
    each row to the true width recorded at propose time; width-0 rows
    become [] and schedule a plain 1-token decode for that request.
    Under structured output the stock D2H rows are real tokens: keep them
    (trimmed to width) so grammar validation still sees actual drafts.
    """
    handler = getattr(runner, "draft_tokens_handler", None)
    if handler is None:
        return False

    # Verify-side TRACE (wake #12): publish exactly what the rejection
    # sampler will consume, from inside the engine process. Closes the
    # last gap between "drafts provably correct" (adapter TRACE 8/8) and
    # "engine accepts 0 of them" — the widths/rows the VERIFY call sees.
    verify_on = os.environ.get("SUFFIX_HYBRID_TRACE_VERIFY", "").strip() != ""
    verify_budget = [int(os.environ.get("SUFFIX_HYBRID_TRACE_VERIFY_MAX",
                                        "12") or 12)]

    def get_draft_tokens(self):
        from vllm.v1.outputs import DraftTokenIds
        if self.draft_tokens_np is not None:
            self.copy_event.synchronize()
            real_rows = self.draft_tokens_np.tolist()
        else:
            real_rows = None
        widths = widths_table() if callable(widths_table) else widths_table
        rows = []
        for i, rid in enumerate(self.req_ids):
            width = int(widths.get(rid, 0)) if isinstance(
                widths, dict) else 0
            if width <= 0 or self.num_draft_tokens <= 0:
                rows.append([])
                continue
            width = min(width, self.num_draft_tokens)
            if real_rows is not None and i < len(real_rows):
                rows.append(real_rows[i][:width])
            else:
                rows.append([-1] * width)
        if verify_on and verify_budget[0] > 0 and any(rows):
            verify_budget[0] -= 1
            print(f"suffix_hybrid VERIFYTRACE rid0={self.req_ids[0]!r} "
                  f"widths={[len(r) for r in rows]} nreqs={len(rows)} "
                  f"ndraft={self.num_draft_tokens}", file=sys.stderr,
                  flush=True)
        return DraftTokenIds(self.req_ids, rows)

    import types
    handler.get_draft_tokens = types.MethodType(get_draft_tokens, handler)
    return True


def _wrap_propose_sync(runner, original, mixer, group, probabilistic=False):
    """Standalone wrapper pinned to the OLD synchronous body (A/B ref)."""
    previous_widths = {}
    state = {"skips": 0, "reason": "", "mixes": 0}
    interval = int(os.environ.get("SUFFIX_HYBRID_LOG_INTERVAL", "0") or 0)
    run = _sync_body(runner, mixer, group, probabilistic, previous_widths,
                     state, interval)

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
        if dummy_run or is_profile or skip_attn_for_dummy_run:
            return native
        try:
            return run(native, input_batch, num_sampled, num_rejected,
                       temperature)
        except Exception as exc:
            state["reason"] = f"{type(exc).__name__}: {exc}"
            state["skips"] += 1
            if state["skips"] <= 3 or state["skips"] % 100 == 0:
                print(f"suffix_hybrid v2 native passthrough (degraded, "
                      f"#{state['skips']}): {state['reason']}",
                      file=sys.stderr, flush=True)
            return native
    return propose


def _wrap_propose(runner, original, mixer, group, probabilistic=False, k=0):
    previous_widths = {}
    state = {"skips": 0, "reason": "", "mixes": 0,
             "skips_unchanged": 0, "skips_probe": 0, "skips_frozen": 0,
             "steps": 0}
    interval = int(os.environ.get("SUFFIX_HYBRID_LOG_INTERVAL", "0") or 0)
    profile = os.environ.get("SUFFIX_HYBRID_PROFILE", "").strip() == "1"
    # Read ONCE at wrapper construction: the per-step zero-op check below is
    # a single closed-over bool, no environ access in the hot path.
    zero_op = os.environ.get("SUFFIX_HYBRID_ZERO_OP", "").strip() == "1"
    # p24-verifierwin: gate wins must be EARNED from the target verifier's
    # verdict (accepted feedback), not from publication-time differences.
    # Self-match rows publish a difference then get rejected; counting those
    # as wins keeps the cooldown gate reset forever under c8 churn (the 58%
    # bench-window share, wakeup #23). Default OFF = p17 semantics.
    verifier_win = os.environ.get(
        "SUFFIX_HYBRID_VERIFIER_WIN", "").strip() == "1"
    hook_times = {"absorb": [0.0, 0], "mix": [0.0, 0],
                  "write": [0.0, 0], "bcast": [0.0, 0]}
    sync_run = _sync_body(runner, mixer, group, probabilistic,
                          previous_widths, state, interval)

    def _bump(name, t0):
        if profile:
            acc = hook_times[name]
            acc[0] += (time.perf_counter() - t0) * 1e3
            acc[1] += 1

    def _degrade_line(kind):
        state["skips"] += 1
        if state["skips"] <= 3 or state["skips"] % 100 == 0:
            print(f"suffix_hybrid v2 native passthrough ({kind}, "
                  f"#{state['skips']}): {state['reason']}",
                  file=sys.stderr, flush=True)

    def _note_skip(kind):
        state[kind] += 1
        n = state[kind]
        if n <= 3 or n % 100 == 0:
            print(f"suffix_hybrid v2 {kind} #{n}",
                  file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Async fast path: one-step-lag CPU mirror + cold/unchanged skips.
    # ------------------------------------------------------------------
    # st["w"] is the optimistic per-step window width K+2 (a step appends at
    # most K verified draft tokens + 1 sampled token). From install k when
    # given, else inferred from the first native draft's row width.
    st = {"mirror": {},             # req_id -> [int] CPU-authoritative row
          "staging": None,          # pinned tail/count/sample buffers
          "cap": 0,
          "w": (int(k) + 2) if int(k or 0) > 0 else 0,
          "stream": None, "prod_ev": None, "ev": None,
          "pending": None,          # None | "SYNC" | [req_id, ...]
          "sync_ids": [],
          "gate": {},               # req_id -> full-path mixes since last win
          "pending_wins": {},       # rid -> True: published-diff row, verdict pending
          "fallback": False, "fallback_logged": False}

    def _tp_mode():
        return os.environ.get("SUFFIX_HYBRID_TP_MODE", "broadcast").strip() \
            or "broadcast"

    def _replay():
        # Replay: every rank mirrors + mixes locally, no collectives at all.
        # Assumes rank-uniform native drafts (the drafter forward is
        # collective over TP); measurement-only, broadcast stays default.
        return _tp_mode() == "replay"

    def _can_skip_collective():
        # Returning without touching the group is only safe when no rank can
        # be left waiting on a broadcast we decided not to send: a
        # single-rank group (no collective exists) or replay (every rank
        # made the same decision from rank-invariant inputs). In multi-rank
        # broadcast mode the slaves' caches are never fed, so they cannot
        # evaluate rank 0's cold/unchanged predicate — keep the (cheap)
        # native broadcast instead of deadlocking their receive.
        return int(getattr(group, "world_size", 1)) <= 1 or _replay()

    def _ensure_staging(n_rows, row_width):
        if st["w"] <= 0:
            st["w"] = max(int(row_width), 1) + 2
        if st["staging"] is not None and st["cap"] >= n_rows:
            return
        cap = max(n_rows, 8)
        # Pinned host buffers make the side-stream D2H truly async; fall
        # back to plain pageable memory when pinning/CUDA is unavailable
        # (CPU-only tests) — the code path is identical.
        pin = bool(torch.cuda.is_available())
        st["staging"] = {
            "tail": torch.empty((cap, st["w"]), dtype=torch.int64,
                                pin_memory=pin),
            "counts": torch.empty(cap, dtype=torch.int64, pin_memory=pin),
            "sampled": torch.empty(cap, dtype=torch.int64, pin_memory=pin),
            "rejected": torch.empty(cap, dtype=torch.int64, pin_memory=pin),
        }
        st["cap"] = cap
        if torch.cuda.is_available() and st["stream"] is None:
            st["stream"] = torch.cuda.Stream()
            st["prod_ev"] = torch.cuda.Event()
            st["ev"] = torch.cuda.Event()
        # A buffer re-grow orphans anything in flight against the old
        # buffers — drop the pending step (mirrors re-seed on demand).
        st["pending"] = None

    def _seed_row(rid, gpu_row, total):
        # One-time-per-request seed: full current history contents into the
        # mirror (plus one scalar total_len read) so the bounded window is
        # positioned at the live end of history from step one. Paid once per
        # request lifetime, never per decode step.
        total = max(int(total), 0)
        upto = min(total, int(gpu_row.shape[0]))
        st["mirror"][rid] = [int(t) for t in gpu_row[:upto].tolist()]

    def _window_slice(gpu_row, L):
        # Bounded optimistic window [L, L+W): never needs a GPU-side length
        # to slice. Production rows are preallocated to max_model_len so the
        # window is always in bounds; narrow fake buffers (tests) clamp
        # instead of raising.
        w = int(gpu_row.shape[0])
        s = min(max(int(L), 0), w)
        return gpu_row[s:min(w, s + st["w"])]

    def _enqueue(ids, idx, num_sampled, num_rejected, k_slots):
        _ensure_staging(max(len(ids), 1), k_slots)
        bufs = st["staging"]
        mirror = st["mirror"]
        ats = runner.req_states.all_token_ids.gpu
        total_gpu = runner.req_states.total_len.gpu
        # Hygiene: rows that left the batch lose their mirror; a later
        # reappearance re-seeds from the authoritative GPU row.
        live = set(ids)
        for rid in [r for r in mirror if r not in live]:
            del mirror[rid]
            st["gate"].pop(rid, None)   # gone request: drop gate state too
            # p24: a pending verdict for a gone request can never resolve
            # (its feedback buffers left the batch with it) — drop it so
            # pending_wins cannot grow unboundedly under churn.
            st["pending_wins"].pop(rid, None)
        cuda = torch.cuda.is_available()
        if cuda:
            # Order the side-stream copies behind the postprocess producer
            # work already queued on the current stream — without this the
            # copies could race and read half-updated rows.
            st["prod_ev"].record()
            ctx = torch.cuda.stream(st["stream"])
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            if cuda:
                st["prod_ev"].wait()
            for i, (rid, j) in enumerate(zip(ids, idx)):
                if rid not in mirror:
                    _seed_row(rid, ats[j], int(total_gpu[j]))
                src = _window_slice(ats[j], len(mirror[rid]))
                if src.numel():
                    bufs["tail"][i, :src.numel()].copy_(src,
                                                        non_blocking=cuda)
                bufs["counts"][i].copy_(total_gpu[j], non_blocking=cuda)
            if ids:
                n = len(ids)
                bufs["sampled"][:n].copy_(num_sampled[:n],
                                          non_blocking=cuda)
                bufs["rejected"][:n].copy_(num_rejected[:n],
                                           non_blocking=cuda)
            if cuda:
                st["ev"].record(st["stream"])
        if cuda:
            st["pending"] = list(ids)
        else:
            # CPU-only (tests / no CUDA): copies are inherently synchronous
            # and already landed in the same pinned-shaped staging — the
            # caller absorbs them inline right after enqueue.
            st["sync_ids"] = list(ids)
            st["pending"] = "SYNC"

    def _absorb_pending():
        if st["pending"] is None:
            return
        bufs = st["staging"]
        pending = st["pending"]
        st["pending"] = None
        if pending == "SYNC":
            ids_now = st["sync_ids"]
        else:
            # One full engine step has passed since record(): this wait is
            # nearly free and is the ONLY host sync in the fast path.
            st["ev"].synchronize()
            ids_now = list(pending)
        if not ids_now:
            return
        counts = bufs["counts"][:len(ids_now)].tolist()
        tails = bufs["tail"][:len(ids_now)].tolist()
        mirror = st["mirror"]
        for rid, new_len, tail_row in zip(ids_now, counts, tails):
            toks = mirror.get(rid)
            if toks is None:
                continue
            last = len(toks)
            new_len = int(new_len)
            if new_len <= last:
                if new_len == 0:
                    # Request reset: drop the key so the next enqueue
                    # re-seeds the whole row from the authoritative GPU
                    # buffer (an emptied list would not retrigger seeding).
                    mirror.pop(rid, None)
                continue
            if new_len - last > st["w"]:
                # Window overshot: same full re-seed path from position 0.
                mirror.pop(rid, None)
                continue
            toks.extend(int(t) for t in tail_row[:new_len - last])

    def _greedy_rows(temperature, idx):
        # Probabilistic mode: arbitrate only temperature==0 rows (the
        # sampler ignores draft_logits there); stochastic rows keep native.
        if not probabilistic:
            return None
        return [t == 0.0 for t in
                temperature[idx].detach().cpu().tolist()]

    def _accepted_lengths(ids, sampled, rejected):
        # Same step-late feedback contract as the sync body: previous_widths
        # describes the PREVIOUS proposal's widths. On CUDA the absorbed
        # sampled/rejected lag one further engine step; fine — this gate
        # only prunes trust in a stale split guess, and every draft is
        # target-verified regardless.
        accepted = []
        for req_id, ns, nr in zip(ids, sampled, rejected):
            a, verified = ns - 1, ns + nr - 1
            previous = previous_widths.get(req_id, 0)
            valid = (ns > 0 and nr >= 0 and 0 < verified <= previous
                     and 0 <= a <= verified
                     and (a < verified or verified == previous))
            accepted.append(a if valid else -1)
        return accepted

    def _check_widths(mixed, k_slots):
        # Fixed-width row semantics identical to the sync body: slots past
        # the proposal keep native tokens; longer-than-K is a mixer bug.
        for i, row in enumerate(mixed):
            if len(row) > k_slots:
                raise ValueError(
                    f"mixed row {i} length {len(row)} exceeds K={k_slots}")

    def _mix_from_mirror(native, ids, idx, temperature):
        mirror = st["mirror"]
        bufs = st["staging"]
        n = len(ids)
        counts_np = np.zeros(max(n, 1), dtype=np.int64)
        for i, rid in enumerate(ids):
            counts_np[i] = len(mirror.get(rid, ()))
        max_len = int(counts_np.max()) if n else 0
        # Pad every mirror row to max_len with zeros: mix_numpy bounds all
        # reads by counts, so the pad is never observed — exactly how
        # vLLM's real preallocated all_token_ids buffer behaves beyond
        # total_len (it holds stale tokens the mixer ignores identically).
        history_np = np.zeros((max(n, 1), max(max_len, 1)), dtype=np.int64)
        for i, rid in enumerate(ids):
            toks = mirror.get(rid)
            if toks:
                history_np[i, :len(toks)] = toks
        # ONE small D2H for the whole step (the native draft rows). Replaces
        # the old clone + full-history blocking buffer read + two small
        # blocking sampler-feedback reads.
        native_rows = native.detach().cpu().tolist()
        sampled = bufs["sampled"][:n].tolist() if n else []
        rejected = bufs["rejected"][:n].tolist() if n else []
        greedy_rows = _greedy_rows(temperature, idx)
        accepted = _accepted_lengths(ids, sampled, rejected)
        if verifier_win and st["pending_wins"]:
            # Resolve last step's published-diff rows with the verifier's
            # verdict (one-step-late feedback): a row that was actually
            # accepted earns the gate reset; a published-but-rejected row
            # was a false win — count it as a miss so the cooldown veto
            # can finally engage under self-match churn.
            gate = st["gate"]
            for i, rid in enumerate(ids):
                if rid in st["pending_wins"]:
                    if accepted[i] == -1:
                        # Unknown feedback (stale split guess / no prior
                        # width): neither win nor miss — keep the counter,
                        # drop the pending marker (verdict never arrives).
                        pass
                    elif accepted[i] >= 1:
                        gate[rid] = 0     # verified win: reset the counter
                    else:
                        gate[rid] = gate.get(rid, 0) + 1  # false win: miss
                    st["pending_wins"].pop(rid, None)
            # rids no longer in the batch (gone mid-verdict): their feedback
            # never arrives; treat as unresolved and drop (no reset).
            gone = set(st["pending_wins"]) - set(ids)
            for rid in gone:
                st["pending_wins"].pop(rid, None)
        mixed = mixer.mix_numpy(ids, counts_np, history_np, native_rows,
                                accepted)
        _check_widths(mixed, native.shape[1])
        return mixed, native_rows, greedy_rows

    def _publish(mixed, native, native_rows, greedy_rows, ids):
        # Rows the temperature gate actually allows us to write.
        eff = [row if (row and (greedy_rows is None or greedy_rows[i]))
               else [] for i, row in enumerate(mixed)]
        # Write-back only on a real difference; a written row is
        # "different" when its prefix disagrees with native (short rows
        # keep native tails, so an all-native mix is a no-op).
        changed = any(row and row != native_rows[i][:len(row)]
                      for i, row in enumerate(eff))
        # ADAPTIVE GATE bookkeeping (p14): count consecutive full-path mixes
        # without a WON row per request. A win is a row whose prefix
        # DISAGREES with native (the same predicate as `changed`) — the
        # mixer's echo path returns the native row verbatim on losing
        # steps, and counting those as wins would never cool down.
        gate = st["gate"]
        pending_wins = st["pending_wins"]
        for i, row in enumerate(eff):
            rid = ids[i]
            if row and row != native_rows[i][:len(row)]:
                if verifier_win:
                    # Published a difference: the verdict arrives next step.
                    # Never reset the counter on publication alone.
                    pending_wins[rid] = True
                else:
                    gate[rid] = 0      # won: reset the miss counter
            else:
                gate[rid] = gate.get(rid, 0) + 1
                pending_wins.pop(rid, None)
        if not changed:
            _note_skip("skips_unchanged")
            # previous_widths keeps the sync-body contract (mixed widths,
            # including full-width native-echo rows).
            previous_widths.clear()
            previous_widths.update(zip(ids, map(len, mixed)))
            if _can_skip_collective():
                return native      # identical on every rank: no collective
            # Broadcast mode, TP>1: the slaves block in this receive every
            # step; publishing native keeps the collective without a clone.
            t = time.perf_counter()
            out = group.broadcast(native, src=0)
            _bump("bcast", t)
            return out
        t = time.perf_counter()
        output = native.clone()
        for i, row in enumerate(eff):
            if row:
                output[i, :len(row)] = torch.tensor(
                    row, dtype=output.dtype, device=output.device)
        _bump("write", t)
        if _replay():
            pass                   # measurement-only: no collective at all
        elif int(getattr(group, "world_size", 1)) > 1:
            t = time.perf_counter()
            output = group.broadcast(output, src=0)
            _bump("bcast", t)
        previous_widths.clear()
        previous_widths.update(zip(ids, map(len, mixed)))
        state["mixes"] += 1
        if interval and state["mixes"] % interval == 0:
            try:
                stats = dict(mixer.get_stats())
            except Exception:
                stats = {}
            if profile:
                for name, (total_ms, count) in hook_times.items():
                    stats[f"hook_{name}_ms"] = (
                        round(total_ms / count, 3) if count else 0.0)
            print("suffix_hybrid_native " + json.dumps(stats, sort_keys=True),
                  file=sys.stderr, flush=True)
        return output

    def _lut_index(input_batch, ids):
        # CPU row indices WITHOUT the old idx_mapping.long().tolist() D2H:
        # req_states.req_id_to_index is the plain CPU dict the engine itself
        # maps req_ids through to BUILD idx_mapping (gpu/model_runner.py:
        # `map(self.req_states.req_id_to_index.__getitem__, req_ids)`), so it
        # is exactly the mapping that D2H produced. Fall back to the tensor
        # read only when the dict is unavailable/incomplete — correctness
        # over speed.
        lut = getattr(runner.req_states, "req_id_to_index", None)
        if lut is not None:
            try:
                return [lut[rid] for rid in ids]
            except KeyError:
                pass
        return input_batch.idx_mapping.long().tolist()

    def _cache_n_gram():
        # The cache's index n-gram order (SUFFIX_HYBRID_INDEX_N, default 8).
        try:
            return int(os.environ.get("SUFFIX_HYBRID_INDEX_N", "8") or 8)
        except ValueError:
            return 8

    def _probe_heartbeat():
        # Full-path mix at least every N steps even when the probe keeps
        # gating: mix_core is the corpus ingestion path (gone-ID eviction +
        # Reset finalize live inside it), so unconditional gating would
        # starve the corpus — the probe would then never fire again.
        try:
            return int(os.environ.get(
                "SUFFIX_HYBRID_PROBE_HEARTBEAT", "8") or 8)
        except ValueError:
            return 8

    def _hb_due(hb, state):
        # True when this step is a forced full-mix heartbeat step (the
        # corpus-ingestion safety net). hb<=0 disables heartbeats.
        return hb > 0 and state["steps"] % hb == 0

    def _probe_cooldown():
        # ADAPTIVE GATE (p14): consecutive full-path mixes without a suffix
        # win after which a request stops forcing the full path (probe can
        # then gate it). 0 disables the adaptive gate entirely.
        try:
            return int(os.environ.get(
                "SUFFIX_HYBRID_PROBE_COOLDOWN", "4") or 4)
        except ValueError:
            return 4

    def _mirror_probe(ids, k_slots):
        # ZERO-SYNC GATE (p12): decide whether suffix evidence exists BEFORE
        # paying the native [n,K] D2H. CPU state only: the one-step-stale
        # mirror tail + the cache's own speculate(). When no row has a
        # strong continuation, publishing native verbatim is byte-identical
        # to what the full arbitrate path would output (mix_core echoes
        # native when the cache has nothing to contribute), so the D2H,
        # mix, compare and (where skippable) collective are all skipped.
        # NEVER gates a cold corpus: cache_tokens()==0 means mix_numpy has
        # not fed anything yet, and skipping it would keep it at zero
        # forever — the 64793d98 self-lock in a new disguise. The
        # heartbeat keeps corpus + estimates fed on long gated stretches;
        # boundary-matching makes skipped steps safe (append-only rows keep
        # the tracked context a strict prefix, so the next ungated mix
        # classifies as Continuing with the accumulated delta).
        # Probe preconditions: mixer must expose cache_tokens() and a
        # suffix_cache handle. Test fakes / future variants without them
        # simply never gate (full path every step — the safe default).
        try:
            if mixer.cache_tokens() == 0:
                return True
        except Exception:
            return True
        hb = _probe_heartbeat()
        if hb > 0 and state["steps"] % hb == 0:
            return True
        cache = getattr(mixer, "suffix_cache", None)
        if cache is None or not ids:
            return True
        try:
            depth = int(os.environ.get(
                "SUFFIX_HYBRID_PROBE_DEPTH", "64") or 64)
            min_len = int(os.environ.get(
                "SUFFIX_HYBRID_PROBE_MIN_LEN", "2") or 2)
        except ValueError:
            depth, min_len = 64, 2
        w = st["w"] or (k_slots + 2)
        n_gram = _cache_n_gram()
        mirror = st["mirror"]
        gate = st["gate"]
        cooldown = _probe_cooldown()
        for rid in ids:
            row = mirror.get(rid)
            if row is None:
                # Un-mirrored (fresh request): full path — the first suffix
                # hit often happens right at prompt-echo time.
                return True
            tail = row[-depth:]
            if len(tail) < n_gram:
                continue        # shorter than the index n-gram: no lookup
            try:
                suffix, _score, _matched = cache.speculate(list(tail), w)
            except Exception:
                return True     # probe itself failed: full path
            if len(suffix) >= min_len:
                # ADAPTIVE GATE (p14): evidence exists, but a request whose
                # last COOLDOWN full-path mixes produced no written suffix
                # row (evidence-but-no-win, the p13 finding) no longer
                # forces arbitration. Wins reset the counter, so a request
                # that starts winning fires again. COOLDOWN<=0 disables.
                if cooldown <= 0 or gate.get(rid, 0) < cooldown:
                    return True
        # Every row probed, no row both evidenced AND trusted: native echo.
        return False

    def _fast_body(native, input_batch, num_sampled, num_rejected,
                   temperature):
        # NO cold early-return: mix_numpy IS the corpus ingestion path
        # (mix_core inserts the observed history on every call), so a
        # cache-size gate placed before it would keep the cache cold
        # forever — nothing ever ingests, so cache_tokens() never leaves
        # zero (live-pod bug, 2026-09-22: skips_cold climbing while
        # suffix_hybrid_native never printed once). The cold mix provably
        # echoes native and _publish's unchanged path then returns native
        # with no clone and no collective where that is safe. The cost
        # this saves us is one small native [n, K] D2H and a bounded
        # window copy per step.
        replay = _replay()
        # Broadcast mode: non-owner ranks contribute nothing but the
        # broadcast receive (they never mix, never mirror — same posture
        # as the sync body; the owner's collective call matches this one
        # on every step regardless of which branch the owner took).
        if group.rank_in_group != 0 and not replay:
            t = time.perf_counter()
            out = group.broadcast(native.clone(), src=0)
            _bump("bcast", t)
            return out
        ids = list(input_batch.req_ids)
        # FROZEN FAST PATH (p17): when the corpus is warm AND every live
        # request has already earned the cooldown veto AND this step is
        # not a heartbeat step, the probe decision tree provably reduces
        # to "gate" (mirror rows only grow, the probe sees the same or a
        # longer tail, wins would have reset the counter). Skipping
        # absorb/enqueue/probe entirely makes cooled steady-state cost
        # ~a dict lookup per step instead of the event sync + copy
        # launches + speculate calls. Identical outputs by construction;
        # new requests (mirror miss -> full path) and heartbeat steps
        # thaw the freeze.
        cooldown = _probe_cooldown()
        hb = _probe_heartbeat()
        try:
            warm = mixer.cache_tokens() > 0
        except Exception:
            warm = False
        gate = st["gate"]
        if (cooldown > 0 and warm
                and all(gate.get(rid, 0) >= cooldown for rid in ids)
                and all(rid in st["mirror"] for rid in ids)):
            # Advance the step counter here so the heartbeat modulo keeps
            # ticking during long frozen stretches — otherwise a lone
            # cooled request would freeze forever and never re-arm via
            # the heartbeat full-mix (the p14 re-arm cycle).
            state["steps"] += 1
            if _hb_due(hb, state):
                # Heartbeat step: undo the pre-increment so the full path's
                # own steps++ lands exactly on the hb multiple and
                # _mirror_probe (which re-checks steps%hb==0) honors the
                # forced full-mix. Without this, the double increment skips
                # the multiple and the corpus never ingests on thaw steps.
                state["steps"] -= 1
            else:
                _note_skip("skips_frozen")
                previous_widths.update(
                    zip(ids, [int(native.shape[1])] * len(ids)))
                return native
        idx_cpu = _lut_index(input_batch, ids)
        t = time.perf_counter()
        # Absorb last step's async copies (one engine step old: the event
        # wait is nearly free), then enqueue this step's window +
        # sampler-feedback copies for NEXT step. The mirror lags at most
        # one step, so the FIRST real step for a request may legitimately
        # pass through native; every draft is target-verified anyway.
        _absorb_pending()
        _enqueue(ids, idx_cpu, num_sampled, num_rejected,
                 int(native.shape[1]))
        if not torch.cuda.is_available():
            _absorb_pending()      # CPU-only: copies already landed
        _bump("absorb", t)
        state["steps"] += 1
        if not _mirror_probe(ids, int(native.shape[1])):
            # No suffix evidence on any row and the corpus is warm: the
            # native echo is byte-identical to the mix's output. Publish
            # verbatim — zero GPU reads, no clone, no mix. Keep the
            # feedback contract: previous_widths records the width we
            # actually published (full-width native rows).
            _note_skip("skips_probe")
            previous_widths.update(
                zip(ids, [int(native.shape[1])] * len(ids)))
            return native
        t = time.perf_counter()
        # Mix purely from the CPU mirror. Greedy mode never touches idx;
        # probabilistic mode needs a tensor for temperature[idx] (CPU
        # tensor built from the LUT — a sync only in probabilistic mode).
        idx_for_greedy = (torch.tensor(idx_cpu, dtype=torch.long)
                          if probabilistic else None)
        mixed, native_rows, greedy_rows = _mix_from_mirror(
            native, ids, idx_for_greedy, temperature)
        _bump("mix", t)
        # Write-back + collective only when the mix actually differs.
        return _publish(mixed, native, native_rows, greedy_rows, ids)

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
        # ZERO-OP ARM (p22): the armed hook does literally nothing per step —
        # no state dict, no environ reads, no timers, no try/except. The
        # log-archaeology residual (~2.7ms/step armed-vs-inert at 8% stall
        # share, p16≈p17 within noise) is not explained by stall, per-step
        # Python, fused-decode, FE or kernels; this arm bounds the residual
        # BY ELIMINATION. If c1 recovers to ~baseline with ZERO_OP=1, the
        # cost is in the per-step machinery and p17's frozen path must get
        # cheaper; if it does NOT recover, the cost is the hook's mere
        # presence (bound-method dispatch, torch dispatcher re-entry on the
        # returned tensor, GC pressure) and the mix must move out of the
        # per-step path entirely (engine patch, not wrapper).
        if zero_op:
            return native
        try:
            if (not st["fallback"]
                    and os.environ.get("SUFFIX_HYBRID_SYNC_HOOK", "").strip()
                    != "1"):
                try:
                    return _fast_body(native, input_batch, num_sampled,
                                      num_rejected, temperature)
                except Exception as exc:
                    # The async machinery failing once is enough: run the
                    # proven synchronous body for the rest of this wrapper's
                    # life instead of crash-looping the fast path.
                    st["fallback"] = True
                    state["reason"] = (
                        f"async hook {type(exc).__name__}: {exc}; "
                        "synchronous fallback from here on")
                    if not st["fallback_logged"]:
                        st["fallback_logged"] = True
                        _degrade_line("fallback")
            return sync_run(native, input_batch, num_sampled, num_rejected,
                            temperature)
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
                      "hook inactive.",
                      file=sys.stderr, flush=True)
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
            if os.environ.get(
                    "SUFFIX_HYBRID_SUFFIX_ONLY", "").strip() == "1":
                if probabilistic:
                    print("suffix_hybrid v2 WARNING: suffix-only mode "
                          "requires greedy draft sampling; staying native.",
                          file=sys.stderr, flush=True)
                    return result
                wrapped = _suffix_only_wrap(
                    self, speculator, mixer, group, k)
                wrapped._suffix_hybrid_hook = True
                speculator.propose = wrapped
                # Ragged per-row widths: patch the CPU-side DraftTokenIds
                # the scheduler consumes (miss rows -> plain 1x decode).
                patched = _patch_get_draft_tokens(
                    self, wrapped._suffix_proposer.widths_table)
                v2 = _wrap_rejection_sampler(self, wrapped, k)
                print(f"suffix_hybrid v2 SUFFIX-ONLY installed "
                      f"speculator={cls_name} k={k} tp={group.world_size} "
                      f"ragged_handler={'yes' if patched else 'NO (uniform)'} "
                      f"vtrace2={'yes' if v2 else 'off'}",
                      file=sys.stderr, flush=True)
                return result
            # Instance-attribute bind: the runner calls
            # self.speculator.propose(input_batch=..., ...) with keywords, so
            # the wrapper closes over the speculator's BOUND class function
            # (self included) plus this runner/mixer/tp-group.
            bound = type(speculator).propose.__get__(speculator,
                                                     type(speculator))
            wrapped = _wrap_propose(self, bound, mixer, group,
                                    probabilistic=probabilistic, k=k)
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
