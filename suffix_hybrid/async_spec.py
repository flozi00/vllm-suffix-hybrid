# SPDX-License-Identifier: Apache-2.0
"""Async-scheduling-compatible suffix / hybrid drafting (SUFFIX_HYBRID_ASYNC=1).

Default OFF. With the gate unset install_v2 never imports this module and
the sync wrap runs exactly as before. Mechanism and vLLM v0.30.0 anchors:
docs/async-spec-v030.md.

Why the sync wrap cannot run under async scheduling: it decides per-row
widths on the host from the CURRENT history (one blocking totals D2H per
step) and hands them to the scheduler through take_draft_token_ids, which
EngineCore.post_step only calls when async scheduling is off
(engine/core.py:621-628).

This path removes both dependencies:

1. Lag-1 host lookup. Every propose(N) enqueues a non-blocking D2H of
   total_len for its rows into pinned memory and records an event. The
   NEXT propose(N+1) synchronizes that event (recorded one step earlier,
   the same pattern as vLLM's AdaptiveVerificationManager.record_confidences,
   adaptive_verification.py:259-262) and runs the Rust suffix lookup on the
   step-N history H_N (UVA rows, fenced by that event). It asks for a
   candidate C of length 2K+1 anchored at p = len(H_N).
2. Device resolve. C, p and len(C) go up with one non-blocking H2D. On
   the GPU, after post_update(N+1), the row's true length t is known, so
   o = t - p (1 + accepted at N+1, in 1..K+1). If all_token_ids[p:t] == C[:o],
   the drafts are C[o:o+K] (only positions < len(C) are real), otherwise
   the row gets filler 0. Filler is always in-vocab and correctness never
   depends on it: every draft is target-verified.
3. Host-exact trim (suffix-only). The host knows an upper bound per row
   before the next step, w = min(K, len(C) - 1). The runner's
   execute_model gets a shallow-copied SchedulerOutput whose spec rows are
   trimmed to w (width 0 = plain decode). The CPU and device query_start_loc
   stay identical, so no attention backend needs
   supports_device_cpu_query_lens_mismatch. The scheduler never learns
   about the trim: it charged 1+K and reconciles through num_rejected =
   K - accepted (scheduler.py:1977-1995), which is exact for the trimmed
   batch. The verify batch shapes are the ones the sync ragged path
   already produces.
   Hybrid mode does not trim. The native drafter fills every row at width K,
   and a row is replaced by the suffix only when the device resolve yields
   >= SUFFIX_HYBRID_ASYNC_HYB_MIN real drafts (default K, so suffix and
   native tails are never stitched).

Invariants:
  I1 The only host wait in propose is Event.synchronize() on the event the
     PREVIOUS propose recorded. There is no torch.cuda.synchronize, no
     blocking copy, and no .item()/.tolist()/.cpu() on a device tensor.
     Under MultiprocExecutor the worker main thread would otherwise idle
     until execute_model(N+2), which the engine sends only after output N
     (whose copy_event precedes this event by one tiny kernel). Under
     UniProc the engine thread waits for that same output right after
     sample_tokens returns.
  I2 The pinned staging is double-buffered by step parity. Every step
     records its event last (in a finally), so the event synchronized at
     the top of step N+2 was recorded on the main stream after every copy
     through slot s at step N had been enqueued.
  I3 Rank invariance. Suffix-only always replays: every rank looks up its
     own (identical) UVA history with a deterministic lag, so host widths
     and drafts match without a collective. Hybrid TP>1 defaults to
     broadcast: rank 0 computes, and the [n, K] rows are broadcast on the
     device, always and from every rank, so a receive is never orphaned.
  I4 Temperature. Suffix-only requires draft_sample_method=greedy
     (draft_logits None: drafts are a point mass, valid at any
     temperature). In hybrid probabilistic mode only temperature==0 rows
     may be replaced.

Env: SUFFIX_HYBRID_ASYNC=1 arms this path (with SUFFIX_HYBRID_WRAP=1).
SUFFIX_HYBRID_SUFFIX_ONLY=1 selects suffix-only, otherwise hybrid.
SUFFIX_HYBRID_ASYNC_TRIM=0 disables the trim, so every row is verified
K-wide with filler (uniform 1+K, which keeps FULL decode graphs).
SUFFIX_HYBRID_ASYNC_HYB_MIN sets the hybrid replace threshold.
SUFFIX_HYBRID_SUFFIX_MIN, SUFFIX_HYBRID_TP_MODE and
SUFFIX_HYBRID_LOG_INTERVAL keep their wrap_v2 meaning.
"""
import contextlib
import copy
import functools
import json
import os
import sys

import numpy as np
import torch

try:  # vLLM's sync checker (VLLM_GPU_SYNC_CHECK); the lag wait is declared.
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
except Exception:  # noqa: BLE001 - CPU tests / other vLLM pins
    gpu_sync_allowed = contextlib.nullcontext


def enabled() -> bool:
    return os.environ.get("SUFFIX_HYBRID_ASYNC", "").strip() == "1"


def _make_event():
    if torch.cuda.is_available():
        return torch.cuda.Event(blocking=True)   # sleep, do not spin
    from suffix_hybrid.wrap_v2 import _PipeSimEvent   # CPU: copies are sync
    return _PipeSimEvent()


def resolve(pack, lc, k, totals, ats, idx):
    """Device-side draft resolve: torch ops only, no host sync.

    pack [n, lc+2] int64 = candidate C[0:lc] | anchor p | len(C).
    totals [n] current total_len of each row (after post_update).
    ats [rows, max_model_len] token history, idx [n] row indices.
    Returns (drafts [n, k] int64 with filler 0, real [n, k] bool).
    """
    cand, p, clen = pack[:, :lc], pack[:, lc], pack[:, lc + 1]
    o = totals.to(torch.int64) - p
    ar = torch.arange(lc, device=pack.device)
    pos = (p[:, None] + ar).clamp(0, ats.shape[1] - 1)
    hist = ats[idx.to(torch.long)[:, None], pos].to(torch.int64)
    unseen = ar[None, :] >= o[:, None]
    aligned = ((hist == cand) | unseen).all(1) & (o >= 1)
    j = o[:, None] + torch.arange(k, device=pack.device)
    real = aligned[:, None] & (j < clen[:, None])
    drafts = cand.gather(1, j.clamp(0, lc - 1)) * real
    return drafts, real


class AsyncSuffixDrafter:
    """One per worker: propose() replaces/wraps speculator.propose, trim()
    rewrites the next SchedulerOutput to the host-known widths."""

    def __init__(self, runner, speculator, proposer, k, native=None,
                 group=None, probabilistic=False, hyb_min=None, interval=0,
                 make_event=_make_event):
        self.k, self.lc = int(k), 2 * int(k) + 1
        self.spec, self.proposer, self.native = speculator, proposer, native
        self.group = group                    # set only in broadcast mode
        self.owner = group is None or group.rank_in_group == 0
        self.probabilistic = probabilistic
        self.hyb_min = self.k if not hyb_min else max(1, min(self.k, hyb_min))
        self.interval = int(interval or 0)
        rs = runner.req_states
        self.ats_np = rs.all_token_ids._uva_buf.np   # host view (UVA)
        self.ats = rs.all_token_ids.gpu
        self.totals = rs.total_len.gpu
        rows, dev = speculator.draft_tokens.shape[0], speculator.draft_tokens.device
        pin = dev.type == "cuda"
        # [rows] lagged totals + [1] cumulative real-draft counter (lag 1).
        self.snap = [torch.zeros(rows + 1, dtype=torch.int64, pin_memory=pin)
                     for _ in range(2)]
        self.pack = [torch.zeros((rows, self.lc + 2), dtype=torch.int64,
                                 pin_memory=pin) for _ in range(2)]
        self.pack_dev = torch.zeros((rows, self.lc + 2), dtype=torch.int64,
                                    device=dev)
        self.real_total = torch.zeros(1, dtype=torch.int64, device=dev)
        self.events = [make_event(), make_event()]
        self.pending = None       # (slot, ids, idx_np, n) of the last propose
        self.last_ev = None       # event recorded at the end of the last step
        self.widths = {}          # rid -> host upper bound for the next verify
        self.step = 0
        self.stats = {"steps": 0, "rows": 0, "cand_rows": 0,
                      "sched_drafts": 0, "real_drafts": 0, "trimmed": 0,
                      "degraded": 0, "reason": ""}

    # ---- propose ----------------------------------------------------------
    def propose(self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens, temperature,
                seeds, dp_sync=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False):
        native = None
        if self.native is not None:
            native = self.native(
                input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=dp_sync,
                dummy_run=dummy_run,
                skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                mm_inputs=mm_inputs, is_profile=is_profile)
        n = int(input_batch.num_reqs)
        if dummy_run or is_profile or skip_attn_for_dummy_run:
            if native is not None:
                return native
            out = self.spec.draft_tokens[:n]
            out.zero_()
            return out
        out = native
        if self.owner:
            try:
                out = self._step(input_batch, native, temperature)
            except Exception as exc:  # noqa: BLE001 - drafts are advisory
                self._degrade(exc)
                if native is None:
                    out = self.spec.draft_tokens[:n]
                    out.zero_()
        if self.group is not None:
            # Every rank, every real step, unconditionally (I3).
            self.group.broadcast(out, src=0)
        return out

    def _absorb(self, ids, idx_np):
        """Rows of the previous propose's snapshot still at the same index.

        Its event was synchronized at the top of _step."""
        if self.pending is None:
            return [], np.zeros(0, dtype=np.int64)
        s, pids, pidx, _ = self.pending
        self.pending = None
        snap = self.snap[s].numpy()
        self.stats["real_drafts"] = int(snap[-1])
        pos = {rid: j for j, rid in enumerate(pids)}
        sel, tot = [], []
        for i, rid in enumerate(ids):
            j = pos.get(rid)
            if j is not None and pidx[j] == idx_np[i] and snap[j] > 0:
                sel.append(i)
                tot.append(int(snap[j]))
        return sel, np.asarray(tot, dtype=np.int64)

    def _step(self, input_batch, native, temperature):
        n = int(input_batch.num_reqs)
        ids = list(input_batch.req_ids)
        idx_np = np.asarray(input_batch.idx_mapping_np, dtype=np.int64)[:n]
        idx = input_batch.idx_mapping[:n]
        s = self.step & 1
        self.step += 1
        if self.last_ev is not None:
            # The only host wait (I1): the previous step's event. It was
            # recorded after every copy through slot s (I2), even when that
            # step degraded and dropped its snapshot.
            with gpu_sync_allowed():
                self.last_ev.synchronize()
        sel, lag_tot = self._absorb(ids, idx_np)
        try:
            return self._lookup_and_resolve(
                native, temperature, s, n, ids, idx_np, idx, sel, lag_tot)
        finally:
            self.events[s].record()
            self.last_ev = self.events[s]

    def _lookup_and_resolve(self, native, temperature, s, n, ids, idx_np,
                            idx, sel, lag_tot):
        k, lc = self.k, self.lc
        snap = self.snap[s]
        snap[:n].copy_(self.totals[idx].to(torch.int64), non_blocking=True)
        snap[-1:].copy_(self.real_total, non_blocking=True)
        self.pending = (s, ids, idx_np.copy(), n)
        # Lag-1 Rust lookup (also ingests rows that left the batch).
        sel_ids = [ids[i] for i in sel]
        packed, clen = self.proposer.propose_suffix_only(
            sel_ids, idx_np[sel], lag_tot, self.ats_np)
        clen = np.asarray(clen, dtype=np.int64)
        pack = self.pack[s][:n]
        pack.zero_()
        if sel:
            pv = pack.numpy()
            pv[sel, :lc] = packed
            pv[sel, lc] = lag_tot
            pv[sel, lc + 1] = clen
        w = np.clip(clen - 1, 0, k)          # o >= 1 -> at most len(C)-1
        self.widths = {rid: int(x) for rid, x in zip(sel_ids, w) if x > 0}
        dev = self.pack_dev[:n]
        dev.copy_(pack, non_blocking=True)
        drafts, real = resolve(dev, lc, k, self.totals[idx], self.ats, idx)
        self.real_total += real.sum()
        st = self.stats
        st["steps"] += 1
        st["rows"] += n
        st["cand_rows"] += len(self.widths)
        st["sched_drafts"] += int(w.sum())
        if self.interval and st["steps"] % self.interval == 0:
            print("suffix_hybrid async " + json.dumps(st, sort_keys=True),
                  file=sys.stderr, flush=True)
        if native is None:
            out = self.spec.draft_tokens[:n]
            out.copy_(drafts)
            return out
        use = real.sum(1) >= self.hyb_min
        if self.probabilistic:
            use &= temperature[idx.to(torch.long)] == 0
        return torch.where(use[:, None], drafts, native)

    def _degrade(self, exc):
        # Widths retracted (trim -> plain decode), lag chain re-seeded.
        self.widths = {}
        self.pending = None
        st = self.stats
        st["degraded"] += 1
        st["reason"] = f"{type(exc).__name__}: {exc}"
        if st["degraded"] <= 3 or st["degraded"] % 100 == 0:
            print(f"suffix_hybrid async passthrough (#{st['degraded']}): "
                  f"{st['reason']}", file=sys.stderr, flush=True)

    # ---- scheduler-output trim (suffix-only) ------------------------------
    def trim(self, so):
        """Shallow copy of `so` with spec rows cut to the host widths.

        Never mutates `so`: under UniProcExecutor it is the scheduler's own
        object, whose update_from_output needs the untrimmed spec lists.
        """
        spec = getattr(so, "scheduled_spec_decode_tokens", None)
        if not spec or getattr(so, "has_structured_output_requests", False):
            return so   # structured output: engine bitmask is K-wide
        new_spec, num, cut = {}, None, 0
        for rid, toks in spec.items():
            w = min(self.widths.get(rid, 0), len(toks))
            if w < len(toks):
                if num is None:
                    num = dict(so.num_scheduled_tokens)
                num[rid] -= len(toks) - w
                cut += len(toks) - w
            if w:
                new_spec[rid] = toks[:w]
        if not cut:
            return so
        so = copy.copy(so)
        so.scheduled_spec_decode_tokens = new_spec
        so.num_scheduled_tokens = num
        so.total_num_scheduled_tokens = so.total_num_scheduled_tokens - cut
        self.stats["trimmed"] += cut
        return so


def install(runner, speculator, k, group, probabilistic):
    """Arm the async drafter on a loaded V2 runner. True when installed."""
    suffix_only = os.environ.get("SUFFIX_HYBRID_SUFFIX_ONLY", "").strip() == "1"
    if suffix_only and probabilistic:
        print("suffix_hybrid async WARNING: suffix-only needs greedy draft "
              "sampling; staying native.", file=sys.stderr, flush=True)
        return False
    if not 1 <= 2 * k + 1 <= 64:
        print(f"suffix_hybrid async WARNING: k={k} too large for the 2k+1 "
              f"candidate (Rust cap 64); staying native.", file=sys.stderr,
              flush=True)
        return False
    vc = getattr(runner, "vllm_config", None)
    pp = getattr(getattr(vc, "parallel_config", None),
                 "pipeline_parallel_size", 1)
    if pp != 1:
        print(f"suffix_hybrid async WARNING: pipeline_parallel_size={pp} "
              f"unsupported; staying native.", file=sys.stderr, flush=True)
        return False
    from suffix_hybrid._native import V2SuffixProposer
    min_len = max(int(os.environ.get("SUFFIX_HYBRID_SUFFIX_MIN", "1") or 1), 1)
    max_len = int(getattr(speculator, "max_model_len", 0) or 0) or 32768
    # Width gate / uniform pallet stay off: the gate reads per-step mirror
    # deltas as accept counts, which the lag-1 mirror would mislabel.
    proposer = V2SuffixProposer(2 * k + 1, max_len, min_len, False, False)
    tp = int(getattr(group, "world_size", 1) or 1)
    replay = (suffix_only or tp == 1 or os.environ.get(
        "SUFFIX_HYBRID_TP_MODE", "broadcast").strip() == "replay")
    native = None
    if not suffix_only:
        native = type(speculator).propose.__get__(speculator, type(speculator))
    drafter = AsyncSuffixDrafter(
        runner, speculator, proposer, k, native=native,
        group=None if replay else group, probabilistic=probabilistic,
        hyb_min=int(os.environ.get("SUFFIX_HYBRID_ASYNC_HYB_MIN", "0") or 0),
        interval=int(os.environ.get("SUFFIX_HYBRID_LOG_INTERVAL", "0") or 0))

    @functools.wraps(type(speculator).propose)
    def propose(*args, **kwargs):
        return drafter.propose(*args, **kwargs)

    propose._suffix_hybrid_hook = True
    propose._suffix_async = drafter
    propose._suffix_proposer = proposer     # warm-start seam (warmstart.py)
    speculator.propose = propose

    trim = suffix_only and os.environ.get(
        "SUFFIX_HYBRID_ASYNC_TRIM", "1").strip() != "0"
    if trim:
        original = runner.execute_model

        @functools.wraps(original)
        def execute_model(scheduler_output, *args, **kwargs):
            if not kwargs.get("dummy_run", False):
                scheduler_output = drafter.trim(scheduler_output)
            return original(scheduler_output, *args, **kwargs)

        runner.execute_model = execute_model
    sc = getattr(vc, "scheduler_config", None)
    print(f"suffix_hybrid v2 ASYNC installed "
          f"mode={'suffix-only' if suffix_only else 'hybrid'} k={k} "
          f"cand={2 * k + 1} tp={tp} "
          f"tp_mode={'replay' if replay else 'broadcast'} trim={trim} "
          f"hyb_min={drafter.hyb_min} async_scheduling="
          f"{getattr(sc, 'async_scheduling', 'ATTR-MISSING')}",
          file=sys.stderr, flush=True)
    return True
