# SPDX-License-Identifier: Apache-2.0
"""Cross-request suffix cache with frequency counts and adaptive draft length.

This is the "self-improving" part of the plugin: unlike vLLM's native n-gram
drafter (which only matches within the current request's own sequence), this
cache keeps a global corpus of *finished* request sequences (previous request
outputs; prompts optionally too) and matches the current context against all
of them. On agentic workloads where requests repeat the same schema
(multi-record JSON, tool-call envelopes, ...), acceptance improves over time
as the corpus accumulates more examples of the pattern.

Pure Python (stdlib only). Accepts any Sequence[int] of token ids — numpy
arrays, torch tensor rows converted via .tolist(), or plain lists.
"""

import logging
import math
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("suffix_hybrid")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


class SuffixCache:
    """Global corpus of finished token sequences + n-gram position index.

    Environment variables (all read once at construction):

    - ``SUFFIX_HYBRID_MAX_CACHED_REQUESTS`` (default 512): FIFO cap on the
      number of finished sequences in the corpus. -1 = unlimited.
    - ``SUFFIX_HYBRID_INDEX_PROMPTS`` (default 0): if 1, also index prompt
      token sequences (not just generated outputs).
    - ``SUFFIX_HYBRID_INDEX_N`` (default 8): n-gram size used as the index
      key. The last `index_n` tokens of the match window are looked up.
    - ``SUFFIX_HYBRID_MAX_POSITIONS_PER_NGRAM`` (default 64): cap on
      positions stored per n-gram key (most recent kept).
    - ``SUFFIX_HYBRID_MAX_TREE_DEPTH`` (default 64): max number of trailing
      context tokens considered as the match window.
    - ``SUFFIX_HYBRID_MAX_SPEC_FACTOR`` (default 1.0): draft length is capped
      at ``factor * match_len + offset`` — never speculate more than the
      evidence supports (same heuristic as vLLM's suffix decoding).
    - ``SUFFIX_HYBRID_MAX_SPEC_OFFSET`` (default 0): additive offset for the
      draft-length cap above.
    - ``SUFFIX_HYBRID_MIN_TOKEN_COUNT`` (default 1): minimum number of times
      a continuation must appear in the corpus after the matched context to
      be proposed.
    - ``SUFFIX_HYBRID_MAX_CANDIDATES`` (default 32): stop back-extension
      search after this many candidate positions (latency guard; this code
      runs every decode step).
    """

    def __init__(self) -> None:
        self.max_cached_requests = _env_int("SUFFIX_HYBRID_MAX_CACHED_REQUESTS", 512)
        self.index_prompts = _env_int("SUFFIX_HYBRID_INDEX_PROMPTS", 0) == 1
        self.index_n = max(1, _env_int("SUFFIX_HYBRID_INDEX_N", 8))
        self.max_positions_per_ngram = max(1, _env_int("SUFFIX_HYBRID_MAX_POSITIONS_PER_NGRAM", 64))
        self.max_tree_depth = max(1, _env_int("SUFFIX_HYBRID_MAX_TREE_DEPTH", 64))
        self.max_spec_factor = _env_float("SUFFIX_HYBRID_MAX_SPEC_FACTOR", 1.0)
        self.max_spec_offset = _env_float("SUFFIX_HYBRID_MAX_SPEC_OFFSET", 0.0)
        self.min_token_count = max(1, _env_int("SUFFIX_HYBRID_MIN_TOKEN_COUNT", 1))
        self.max_candidates = max(1, _env_int("SUFFIX_HYBRID_MAX_CANDIDATES", 32))
        self.time_budget_s = _env_float("SUFFIX_HYBRID_TIME_BUDGET_MS", 2.0) / 1000.0
        self._time_warned = False

        self._lock = threading.Lock()
        # Finished sequences: list of List[int]. Entries are never mutated
        # after being appended (eviction only pops from the front).
        self._sequences: List[List[int]] = []
        # n-gram tuple -> list of (seq_idx, position). Built incrementally.
        self._index: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
        # seq_idx -> token position already indexed up to (exclusive).
        self._indexed_upto: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def _index_ngrams(self, seq_idx: int, tokens: List[int], start: int) -> None:
        """Index all n-grams of `tokens` from position `start` onward."""
        n = self.index_n
        for pos in range(start, max(start, len(tokens) - n + 1)):
            key = tuple(tokens[pos:pos + n])
            bucket = self._index.get(key)
            if bucket is None:
                self._index[key] = bucket = []
            bucket.append((seq_idx, pos))
            if len(bucket) > self.max_positions_per_ngram:
                del bucket[: len(bucket) - self.max_positions_per_ngram]

    def _evict_if_needed(self) -> None:
        cap = self.max_cached_requests
        if cap < 0 or len(self._sequences) <= cap:
            return
        overflow = len(self._sequences) - cap
        # FIFO: drop the oldest sequences. Keys referencing dropped seq
        # indices are pruned lazily during lookup (idx >= base check below).
        self._seq_base_idx = getattr(self, "_seq_base_idx", 0) + overflow
        del self._sequences[:overflow]

    _seq_base_idx = 0  # absolute index of self._sequences[0]

    def add_sequence(self, tokens: Sequence[int]) -> None:
        """Add a finished sequence (a request's generated output) to the corpus."""
        if tokens is None:
            return
        toks = list(tokens)
        if len(toks) < self.index_n + 1:
            return
        with self._lock:
            seq_idx = self._seq_base_idx + len(self._sequences)
            self._sequences.append(toks)
            self._index_ngrams(seq_idx, toks, 0)
            self._evict_if_needed()

    def add_prompt(self, tokens: Sequence[int]) -> None:
        """Optionally index prompt tokens (env-toggled, off by default)."""
        if not self.index_prompts:
            return
        self.add_sequence(tokens)

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

    def speculate(
        self,
        context_tokens: Sequence[int],
        max_tokens: int,
    ) -> Tuple[List[int], float, int]:
        """Propose a continuation for `context_tokens`.

        Returns ``(token_ids, score, match_len)``.

        - The last up-to-``max_tree_depth`` tokens of ``context_tokens`` form
          the match window.
        - The ``index_n``-gram at the end of the window is looked up in the
          position index; each candidate position is back-extended as far as
          possible (classic prompt-lookup style). Longest back-extension
          wins; ties are broken by frequency (number of candidates sharing
          the same back-extension endpoint).
        - Draft length is capped by ``max_spec_factor * match_len +
          max_spec_offset`` and by ``max_tokens``.
        - A continuation must appear at least ``min_token_count`` times
          after the matched context (with back-extension ties counted, the
          winner's candidate count is the support; below the threshold we
          return an empty draft).
        - ``score`` is the product of per-token empirical frequencies — the
          fraction of surviving candidates whose sequence continues with each
          proposed token in turn. This is an acceptance-probability estimate
          the hybrid proposer uses to compare suffix vs n-gram drafts.
          (Chosen over a pure match_len-based score because it discounts
          drafts that continue a long match with rare tokens.)
        - Latency guard: at most ``max_candidates`` candidate positions are
          examined, and a soft ~2ms budget per call is respected; a one-time
          warning is logged if exceeded.

        The approximation on support/frequency: counts are computed over the
        candidate positions found in the index bucket only (capped at
        ``max_positions_per_ngram``), not over the full corpus — good enough
        for arbitration and bounded in work.
        """
        started = time.perf_counter()
        window = list(context_tokens)
        if len(window) > self.max_tree_depth:
            window = window[-self.max_tree_depth:]
        if max_tokens <= 0 or len(window) < self.index_n:
            return [], 0.0, 0

        key = tuple(window[-self.index_n:])
        deadline = started + self.time_budget_s

        with self._lock:
            bucket = self._index.get(key)
            if not bucket:
                self._check_budget(started)
                return [], 0.0, 0
            base = self._seq_base_idx
            candidates = [(idx, pos) for (idx, pos) in bucket if idx >= base]
            if not candidates:
                self._check_budget(started)
                return [], 0.0, 0
            candidates = candidates[-self.max_candidates:]

            # Back-extend each candidate; track best (longest) extension.
            # best: match_len -> {endpoint: count}
            ext_counts: Dict[int, Dict[Tuple[int, int], int]] = {}
            for seq_idx, pos in candidates:
                li = seq_idx - base
                if li >= len(self._sequences):
                    continue
                seq = self._sequences[li]
                if seq[pos:pos + self.index_n] != list(key):
                    continue  # stale index entry (shouldn't happen)
                # Extend backwards within both `seq` and `window`.
                ext = 0
                max_ext = min(pos, len(window) - self.index_n)
                while ext < max_ext and seq[pos - ext - 1] == window[-self.index_n - ext - 1]:
                    ext += 1
                    if time.perf_counter() > deadline:
                        break
                endpoint = (seq_idx, pos)
                d = ext_counts.setdefault(ext, {})
                d[endpoint] = d.get(endpoint, 0) + 1
                if time.perf_counter() > deadline:
                    break

            if not ext_counts:
                self._check_budget(started)
                return [], 0.0, 0

            best_len = max(ext_counts.keys())
            endpoints = ext_counts[best_len]

            # Group the tied candidates by the token that follows the match.
            # "Frequency" = how many candidates back each continuation token.
            # Highest count wins; ties broken by most recent sequence.
            contig = {}
            for (si, po), c in endpoints.items():
                s = self._sequences[si - base]
                q = po + self.index_n
                t = s[q] if q < len(s) else None
                contig.setdefault(t, []).append(((si, po), c))
            best_group = max(
                contig.items(),
                key=lambda kv: (sum(c for _, c in kv[1]), kv[1][-1][0][0]),
            )
            token_count = sum(c for _, c in best_group[1])
            if token_count < self.min_token_count:
                # Continuation does not have the minimum required support.
                self._check_budget(started)
                return [], 0.0, self.index_n + best_len

            winner = max(best_group[1], key=lambda ec: ec[0][0])[0]
            seq_idx, pos = winner
            seq = self._sequences[seq_idx - base]
            match_len = self.index_n + best_len
            follow_start = pos + self.index_n

            # Draft-length cap: evidence-based heuristic (vLLM suffix decoding).
            cap = int(self.max_spec_factor * match_len + self.max_spec_offset)
            n_draft = min(max_tokens, cap, len(seq) - follow_start)
            if n_draft <= 0:
                self._check_budget(started)
                return [], 0.0, match_len

            # Score: product of per-token empirical frequencies. Tokens are
            # counted only while every surviving candidate still agrees,
            # so the score reflects the probability the whole prefix is
            # accepted (approximately).
            remaining = {e: c for e, c in endpoints.items()}
            tokens_out: List[int] = []
            score = 1.0
            step = 0
            while step < n_draft and remaining:
                p = follow_start + step
                tok = seq[p] if p < len(seq) else None
                if tok is None:
                    break
                counts: Dict[int, int] = {}
                total = 0
                for (si, po), c in list(remaining.items()):
                    s = self._sequences[si - base]
                    q = po + self.index_n + step
                    t = s[q] if q < len(s) else None
                    counts[t] = counts.get(t, 0) + c
                    total += c
                if total <= 0 or counts.get(tok, 0) < self.min_token_count:
                    break
                score *= counts[tok] / float(total)
                # keep only candidates that continue to agree
                remaining = {
                    (si, po): c
                    for (si, po), c in remaining.items()
                    if (po + self.index_n + step < len(self._sequences[si - base])
                        and self._sequences[si - base][po + self.index_n + step] == tok)
                }
                tokens_out.append(tok)
                step += 1
                if time.perf_counter() > deadline:
                    break

            self._check_budget(started)
            if not tokens_out:
                return [], 0.0, match_len
            return tokens_out, score, match_len

    def _check_budget(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        if elapsed > self.time_budget_s and not self._time_warned:
            self._time_warned = True
            logger.warning(
                "suffix_hybrid: speculate() exceeded soft time budget of %.1f ms "
                "(took %.1f ms). Consider lowering SUFFIX_HYBRID_MAX_CANDIDATES "
                "or SUFFIX_HYBRID_MAX_POSITIONS_PER_NGRAM.",
                self.time_budget_s * 1000.0,
                elapsed * 1000.0,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "num_sequences": len(self._sequences),
                "num_index_keys": len(self._index),
                "index_n": self.index_n,
                "max_cached_requests": self.max_cached_requests,
            }