# SPDX-License-Identifier: Apache-2.0
"""Capture of the drafter's inputs into a bounded pinned host ring.

Hook point: ``speculator.propose`` (installed on the INSTANCE after
load_model, same seam as wrap_v2). v0.30.0 calls it positionally from
model_runner.sample_tokens (~2143) AFTER postprocess_sampled, with
``last_hidden_states`` = the rows the drafter consumes (Qwen4Exp: the
pre-final-mixer multi stream from get_mtp_target_hidden_states), plus
``input_batch`` (host: req_ids, query_start_loc_np, num_computed_tokens_np,
prefill_len_np; device: input_ids, positions) and ``num_rejected`` (device).
The drafter pairs hidden h_p with token x_{p+1}; rows p of a request are
valid for the first ``q_len - num_rejected`` query rows.

Per step, for sampled requests only (deterministic crc32(req_id) rate) and
rank 0 only (TP ranks see identical hidden rows; training broadcasts):
  main stream (device ops only, no host sync): index_select the rows'
  hidden / input_ids / positions / num_rejected into a per-slot DEVICE
  staging buffer (so the next target step may overwrite its own buffers
  freely), then record an event;
  side stream: wait that event, D2H straight into the reserved rows of the
  pinned ARENA (no host memcpy later), record the slot's done event.
After the inner propose returns, the served depth-1 draft of each sampled
request goes the same way (parity metric). Two slots rotate; when both are
still in flight the step is DROPPED (counted), never waited for.
The next steps ``absorb`` completed slots with ``event.query()`` (never
synchronize) and join rows per request by position continuity: labels are
the request's own later tokens (x_{p+1}, x_{p+2}, ... = the input ids of
the following rows), so a window is emitted once L = window + k + 1
consecutive rows exist (the trailing k+1 rows start the next window) or,
shorter (>= min_rows), when the request leaves the batch / has a gap.

Arena = ring of rows with a monotonic write counter; a row (and every
window touching it) is alive while counter >= written - capacity: FIFO
eviction by construction, O(1). Prompt rows are kept only for the last
``ctx`` prompt positions (attention context); only anchors whose next
token was generated (pos >= prefill_len - 1) carry loss / metrics.
"""
from __future__ import annotations

import collections
import inspect
import sys
import zlib

import numpy as np
import torch


class Arena:
    def __init__(self, cap_rows, width, dtype=torch.bfloat16, pin=False):
        self.cap, self.written = int(cap_rows), 0
        self.h = torch.empty(self.cap, width, dtype=dtype, pin_memory=pin)
        self.tok = torch.empty(self.cap, dtype=torch.int32, pin_memory=pin)
        self.pos = torch.empty(self.cap, dtype=torch.int64, pin_memory=pin)
        self.d1 = torch.full((self.cap,), -1, dtype=torch.int32)

    def reserve(self, n):
        """Counter of n contiguous ring rows (the ring tail is skipped rather
        than split); None when n exceeds the ring."""
        if n > self.cap:
            return None
        s = self.written % self.cap
        if s + n > self.cap:
            self.written += self.cap - s
        start = self.written
        self.written += n
        i = start % self.cap
        self.d1[i:i + n] = -1
        return start

    def alive(self, counter):
        return counter >= self.written - self.cap

    def idx(self, counters):
        return torch.as_tensor(np.asarray(counters) % self.cap, dtype=torch.long)


class _Req:
    __slots__ = ("rows", "next_pos", "plen")

    def __init__(self, plen):
        self.rows, self.next_pos, self.plen = [], None, plen


class Store:
    """Rank-0 capture state. `window` = anchors per window."""

    def __init__(self, width, k, cap_rows, window=256, rate=1.0, ctx=64,
                 min_rows=32, max_rows_per_step=2048, heldout_pct=20,
                 dtype=torch.bfloat16, device="cpu"):
        self.k, self.rate, self.ctx = k, rate, ctx
        self.L = window + k + 1
        self.min_rows = max(min_rows, k + 3)
        self.max_rows, self.heldout_pct = max_rows_per_step, heldout_pct
        cuda = torch.device(device).type == "cuda"
        self.arena = Arena(cap_rows, width, dtype, pin=cuda)
        self.device, self.cuda = torch.device(device), cuda
        self.reqs = {}
        self.train, self.held = collections.deque(), collections.deque()
        self.inflight, self.free = collections.deque(), []
        for _ in range(2):
            self.free.append(self._slot(width, dtype))
        self.stream = torch.cuda.Stream(self.device) if cuda else None
        self.stats = collections.Counter()

    def _slot(self, width, dtype):
        dev, pin, n = self.device, self.cuda, self.max_rows
        return dict(h=torch.empty(n, width, dtype=dtype, device=dev),
                    tok=torch.empty(n, dtype=torch.int32, device=dev),
                    pos=torch.empty(n, dtype=torch.int64, device=dev),
                    sel=torch.empty(n, dtype=torch.int64, pin_memory=pin),
                    idx=torch.empty(n, dtype=torch.int64, pin_memory=pin),
                    rej_d=torch.empty(n, dtype=torch.int64, device=dev),
                    rej=torch.empty(n, dtype=torch.int64, pin_memory=pin),
                    d1_d=torch.full((n,), -1, dtype=torch.int32, device=dev),
                    d1=torch.full((n,), -1, dtype=torch.int32, pin_memory=pin),
                    ev=None, prod=None, entries=None, start=0, total=0)

    def selected(self, rid):
        return zlib.crc32(str(rid).encode()) % 10000 < self.rate * 10000

    def heldout(self, rid):
        return zlib.crc32(b"h" + str(rid).encode()) % 100 < self.heldout_pct

    # ------------------------------------------------------------------ step
    def enqueue(self, ib, hidden, num_rejected):
        """Stage this step's rows; returns the slot (for drafts) or None."""
        n_req = int(ib.num_reqs)
        qsl = ib.query_start_loc_np
        comp, plen = ib.num_computed_tokens_np, ib.prefill_len_np
        entries, src, total = [], [], 0
        for i, rid in enumerate(ib.req_ids[:n_req]):
            if not self.selected(rid):
                continue
            qs, qe = int(qsl[i]), int(qsl[i + 1])
            skip = max(0, min(qe - qs, int(plen[i]) - self.ctx - int(comp[i])))
            rows = qe - qs - skip
            if rows <= 0:
                continue
            if total + rows > self.max_rows:
                self.stats["rows_over_budget"] += rows
                continue
            entries.append((rid, i, total, rows, int(plen[i])))
            src.append(np.arange(qs + skip, qe))
            total += rows
        if not entries:
            return None
        if not self.free:
            self.stats["dropped_steps"] += 1
            return None
        start = self.arena.reserve(total)
        if start is None:
            return None
        slot = self.free.pop()
        slot.update(entries=entries, start=start, total=total)
        m = len(entries)
        slot["idx"][:total].copy_(torch.from_numpy(np.concatenate(src)))
        slot["sel"][:m].copy_(torch.tensor([e[1] for e in entries]))
        nb = self.cuda
        idx = slot["idx"][:total].to(self.device, non_blocking=nb)
        sel = slot["sel"][:m].to(self.device, non_blocking=nb)
        pos = ib.positions if ib.positions.dim() == 1 else ib.positions[0]
        slot["h"][:total].copy_(hidden.index_select(0, idx))
        slot["tok"][:total].copy_(ib.input_ids.index_select(0, idx))
        slot["pos"][:total].copy_(pos.index_select(0, idx))
        slot["rej_d"][:m].copy_(num_rejected.index_select(0, sel))
        slot["d1_d"][:m].fill_(-1)
        slot["sel_d"] = sel
        self.inflight.append(slot)
        self.stats["rows"] += total
        return slot

    def enqueue_drafts(self, slot, drafts):
        if slot is not None and drafts is not None and drafts.dim() == 2:
            m = len(slot["entries"])
            slot["d1_d"][:m].copy_(drafts.index_select(0, slot["sel_d"])[:, 0])

    def flush(self, slot):
        """Issue the slot's D2H into the arena (side stream on CUDA)."""
        if slot is None:
            return
        a, s, t, m = self.arena, slot["start"] % self.arena.cap, slot["total"], len(slot["entries"])
        pairs = ((a.h[s:s + t], slot["h"][:t]), (a.tok[s:s + t], slot["tok"][:t]),
                 (a.pos[s:s + t], slot["pos"][:t]), (slot["rej"][:m], slot["rej_d"][:m]),
                 (slot["d1"][:m], slot["d1_d"][:m]))
        if not self.cuda:
            for dst, srcv in pairs:
                dst.copy_(srcv)
            return
        prod = torch.cuda.Event()
        prod.record()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(prod)
            for dst, srcv in pairs:
                dst.copy_(srcv, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        slot["ev"] = ev

    def absorb(self):
        """Join every completed slot (FIFO, event.query only)."""
        while self.inflight:
            slot = self.inflight[0]
            if slot["ev"] is not None and not slot["ev"].query():
                return
            self.inflight.popleft()
            a = self.arena
            if a.alive(slot["start"]):
                for j, (rid, _, off, rows, plen) in enumerate(slot["entries"]):
                    valid = max(0, rows - int(slot["rej"][j]))
                    if valid:
                        c0 = slot["start"] + off
                        d1 = int(slot["d1"][j])
                        if d1 >= 0:
                            a.d1[(c0 + valid - 1) % a.cap] = d1
                        self._append(rid, plen, c0, valid)
            slot["ev"] = None
            self.free.append(slot)

    def depart(self, live_ids):
        """Finalize requests absent from the current batch."""
        live = set(live_ids)
        live.update(e[0] for s in self.inflight for e in s["entries"])
        for rid in [r for r in self.reqs if r not in live]:
            r = self.reqs.pop(rid)
            self._emit(rid, r.rows, r.plen, final=True)

    # -------------------------------------------------------------- windows
    def _append(self, rid, plen, c0, n):
        r = self.reqs.get(rid)
        first = int(self.arena.pos[c0 % self.arena.cap])
        if r is None or r.next_pos != first:
            if r is not None:
                self.stats["gaps"] += 1
                self._emit(rid, r.rows, r.plen, final=True)
            r = self.reqs[rid] = _Req(plen)
        r.rows.extend(range(c0, c0 + n))
        r.next_pos = first + n
        while len(r.rows) >= self.L:
            self._emit(rid, r.rows[:self.L], r.plen, final=False)
            r.rows = r.rows[self.L - (self.k + 1):]

    def _emit(self, rid, rows, plen, final):
        if len(rows) < (self.min_rows if final else self.L):
            return
        dq = self.held if self.heldout(rid) else self.train
        dq.append(dict(rows=np.asarray(rows, dtype=np.int64), plen=plen, rid=rid))
        self.stats["windows_held" if dq is self.held else "windows_train"] += 1
        while dq and not self.arena.alive(int(dq[0]["rows"][0])):
            dq.popleft()
            self.stats["evicted_windows"] += 1

    def materialize(self, w):
        """Window record -> dict of CPU tensors, or None if evicted."""
        if not self.arena.alive(int(w["rows"][0])):
            return None
        a, i = self.arena, self.arena.idx(w["rows"])
        pos = a.pos[i].clone()
        return dict(h=a.h[i].clone(), tok=a.tok[i].clone(), pos=pos, d1=a.d1[i].clone(),
                    anchor=pos >= w["plen"] - 1, rid=w["rid"])

    def alive_windows(self, heldout):
        dq = self.held if heldout else self.train
        return [w for w in dq if self.arena.alive(int(w["rows"][0]))]


# ---------------------------------------------------------------------------
# propose wrapper
# ---------------------------------------------------------------------------
def _capturing():
    try:
        return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
    except Exception:
        return True


def wrap_propose(inner, store, on_error=None):
    """speculator.propose wrapper: capture around the untouched native call.
    Any capture failure disables capture (logged once) and serving goes on."""
    sig = inspect.signature(inner)
    state = {"on": True}

    def _fail(exc):
        state["on"] = False
        print(f"[suffix mtp-tune] capture disabled: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        if on_error is not None:
            on_error(exc)

    def propose(*args, **kwargs):
        slot = None
        if state["on"]:
            try:
                b = sig.bind(*args, **kwargs).arguments
                if not (b.get("dummy_run") or b.get("is_profile") or _capturing()):
                    ib = b["input_batch"]
                    store.absorb()
                    store.depart(ib.req_ids[:ib.num_reqs])
                    slot = store.enqueue(ib, b["last_hidden_states"], b["num_rejected"])
            except Exception as exc:  # noqa: BLE001 - never break serving
                _fail(exc)
                slot = None
        out = inner(*args, **kwargs)
        if slot is not None:
            try:
                store.enqueue_drafts(slot, out)
                store.flush(slot)
            except Exception as exc:  # noqa: BLE001
                _fail(exc)
        return out

    propose._mtp_tune_hook = True
    return propose
