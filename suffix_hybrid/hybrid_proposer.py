# SPDX-License-Identifier: Apache-2.0
"""HybridProposer: suffix cache + native n-gram drafting for vLLM.

Main integration (custom_class mode)::

    --speculative-config '{"method":"custom_class",
                           "model":"suffix_hybrid.hybrid_proposer.HybridProposer",
                           "num_speculative_tokens":8}'

vLLM's ``custom_class`` proposer (``vllm/v1/spec_decode/custom_class_proposer.py``)
imports the module path from ``speculative_config.model`` and instantiates the
class with the ``VllmConfig``; it requires a callable ``.propose``. In this mode
vLLM sets ``prompt_lookup_min=0`` / ``prompt_lookup_max=0`` and loads NO draft
model, so this plugin must NOT rely on those config values — it takes its own
n-gram bounds from env vars (``SUFFIX_HYBRID_NGRAM_MIN`` / ``SUFFIX_HYBRID_NGRAM_MAX``).

vllm is imported lazily inside ``__init__`` behind try/except so unit tests
run without vllm/torch installed.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from suffix_hybrid.suffix_cache import SuffixCache, _env_int

logger = logging.getLogger("suffix_hybrid")


class HybridProposer:
    """Per row: suffix-cache draft vs native n-gram draft; higher score wins.

    Environment variables:

    - ``SUFFIX_HYBRID_USE_NGRAM`` (default 1): also compute vLLM's native
      in-request n-gram draft and pick the better draft per row. The native
      :class:`~vllm.v1.spec_decode.ngram_proposer.NgramProposer` is reused
      (constructed lazily); if its API drifts or it cannot be imported, we
      silently fall back to suffix-only.
    - ``SUFFIX_HYBRID_NGRAM_MIN`` (default 4) / ``SUFFIX_HYBRID_NGRAM_MAX``
      (default 16): min/max n-gram sizes for the native n-gram drafter
      (cannot come from speculative_config, which is zeroed in custom_class
      mode).
    - ``SUFFIX_HYBRID_MAX_SPEC_TOKENS`` (default 0 = use
      ``speculative_config.num_speculative_tokens``): per-row draft cap.
    - ``SUFFIX_HYBRID_STATS_FILE``: if set, append a JSON line every
      ``SUFFIX_HYBRID_STATS_INTERVAL`` (default 100) propose calls with
      ``{ts, proposed, accepted, suffix_proposals, ngram_proposals,
      avg_match_len}``.
    - See :class:`~suffix_hybrid.suffix_cache.SuffixCache` for cache tuning.

    Acceptance-rate telemetry (approximation): the next step's
    ``sampled_token_ids`` ARE the accepted tokens. We count, per request,
    how many leading tokens of our previous proposal appear at the head of
    this step's sampled ids — an approximation (a rejected draft token can
    coincidentally match the sampled id), documented as such.
    """

    def __init__(self, vllm_config: Any = None) -> None:
        self.vllm_config = vllm_config
        self.use_ngram = _env_int("SUFFIX_HYBRID_USE_NGRAM", 1) == 1
        self.ngram_min = _env_int("SUFFIX_HYBRID_NGRAM_MIN", 4)
        self.ngram_max = _env_int("SUFFIX_HYBRID_NGRAM_MAX", 16)
        self.stats_file = os.environ.get("SUFFIX_HYBRID_STATS_FILE", "").strip() or None
        self.stats_interval = max(1, _env_int("SUFFIX_HYBRID_STATS_INTERVAL", 100))
        self.log_interval = max(1, _env_int("SUFFIX_HYBRID_LOG_INTERVAL", 1000))

        self.num_speculative_tokens = 8
        self.max_model_len = 1 << 30
        try:
            spec = getattr(vllm_config, "speculative_config", None)
            if spec is not None:
                self.num_speculative_tokens = int(
                    getattr(spec, "num_speculative_tokens", 8) or 8
                )
            mc = getattr(vllm_config, "model_config", None)
            if mc is not None and getattr(mc, "max_model_len", None):
                self.max_model_len = int(mc.max_model_len)
        except Exception:
            pass
        cap = _env_int("SUFFIX_HYBRID_MAX_SPEC_TOKENS", 0)
        if cap > 0:
            self.num_speculative_tokens = cap

        self.suffix_cache = SuffixCache()

        # request id -> dict(prev_draft=[int], active_tokens=[int])
        self._active: Dict[Any, Dict[str, Any]] = {}
        # Telemetry
        self._calls = 0
        self._proposed = 0
        self._accepted = 0
        self._suffix_proposals = 0
        self._ngram_proposals = 0
        self._match_len_sum = 0.0
        self._match_len_count = 0

        self._ngram_proposer = None
        self._ngram_failed = False

    # ------------------------------------------------------------------

    def _get_ngram_proposer(self) -> Optional[Any]:
        """Lazily construct vLLM's native NgramProposer with patched config."""
        if not self.use_ngram or self._ngram_failed:
            return None
        if self._ngram_proposer is not None:
            return self._ngram_proposer
        try:
            from vllm.config import VllmConfig  # noqa: F401
            from vllm.v1.spec_decode.ngram_proposer import NgramProposer

            cfg = _PatchedConfig(
                self.vllm_config, self.ngram_min, self.ngram_max,
                self.num_speculative_tokens, self.max_model_len,
            )
            self._ngram_proposer = NgramProposer(cfg)
        except Exception as exc:  # API drift or vllm not installed
            self._ngram_failed = True
            logger.info("suffix_hybrid: native ngram proposer unavailable (%s); "
                        "falling back to suffix-only.", exc)
        return self._ngram_proposer

    # ------------------------------------------------------------------

    def propose(self, *args: Any, **kwargs: Any) -> List[List[int]]:
        """Introspected-signature propose().

        Two call shapes exist across vLLM versions:

        - Older / suffix-style:
          ``propose(num_speculative_tokens, input_batch, sampled_token_ids,
          slot_mappings=...)`` — an InputBatch object with ``req_ids``.
        - vLLM >= 0.29 custom_class (verified against v0.29.0
          gpu_model_runner): ``propose(sampled_token_ids,
          num_tokens_no_spec, token_ids_cpu, slot_mappings=...)`` — NO
          InputBatch, so no request ids and no prompt lengths. We detect
          the parts (int -> token cap; ``req_ids`` attr -> input batch;
          list-of-lists -> sampled ids; 1-D list-of-ints/tensor ->
          num_tokens_no_spec; 2-D tensor -> token_ids_cpu) and, in the
          second shape, wrap them in a ``_RowAdapter`` that synthesizes
          per-row request ids (see its docstring for the reuse-detection
          invariant).
        """
        num_spec = self.num_speculative_tokens
        input_batch = None
        sampled_token_ids: List[List[int]] = []
        token_cpu: Any = None
        num_tokens_no_spec_arg: Any = None

        for a in args:
            if isinstance(a, int):
                num_spec = a
            elif hasattr(a, "req_ids") and not isinstance(a, (list, tuple)):
                input_batch = a
            elif isinstance(a, (list, tuple)) and (
                not a or isinstance(a[0], (list, tuple))
            ):
                sampled_token_ids = a
            elif isinstance(a, (list, tuple)) and a and isinstance(a[0], int):
                num_tokens_no_spec_arg = a
            elif _shape_len(a) == 2:
                token_cpu = a
            elif _shape_len(a) == 1:
                num_tokens_no_spec_arg = a
        if "input_batch" in kwargs and kwargs["input_batch"] is not None:
            input_batch = kwargs["input_batch"]
        if kwargs.get("sampled_token_ids") is not None:
            sampled_token_ids = kwargs["sampled_token_ids"]
        if kwargs.get("num_speculative_tokens") is not None and isinstance(
            kwargs["num_speculative_tokens"], int
        ):
            num_spec = kwargs["num_speculative_tokens"]

        if input_batch is None and token_cpu is not None:
            # vLLM >= 0.29 custom_class shape: no InputBatch, positional
            # (sampled_token_ids, num_tokens_no_spec, token_ids_cpu). The
            # adapter PERSISTS on the proposer (self._row_adapter) so row
            # continuity is tracked across steps — a fresh adapter every
            # call would treat every row as new every step and never build
            # request state.
            adapter = getattr(self, "_row_adapter", None)
            if adapter is None:
                adapter = _RowAdapter(
                    token_cpu, num_tokens_no_spec_arg, len(sampled_token_ids)
                )
                self._row_adapter = adapter
            else:
                adapter.update(
                    token_cpu, num_tokens_no_spec_arg, len(sampled_token_ids)
                )
            adapter.refresh(sampled_token_ids)
            input_batch = adapter

        if input_batch is None:
            return [[] for _ in sampled_token_ids]

        self._calls += 1
        drafts: List[List[int]] = []
        chosen_this_step: List[str] = []
        match_lens: List[int] = []

        req_ids = list(input_batch.req_ids)
        num_tokens_no_spec = input_batch.num_tokens_no_spec
        for i, sampled_ids in enumerate(sampled_token_ids):
            req_id = req_ids[i]
            state = self._active.get(req_id)
            if state is not None and state.get("prev_draft") and sampled_ids:
                # Telemetry: how many leading tokens of the previous draft
                # were accepted (approximation: compare against the head of
                # this step's sampled ids).
                prev = state["prev_draft"]
                acc = 0
                for j, tok in enumerate(prev):
                    if j < len(sampled_ids) and sampled_ids[j] == tok:
                        acc += 1
                    else:
                        break
                self._accepted += acc
                self._proposed += len(prev)

            if not sampled_ids:
                # Partial prefill: no speculation for this row.
                drafts.append([])
                continue

            num_tokens = int(num_tokens_no_spec[i]) if i < len(num_tokens_no_spec) else 0
            if num_tokens >= self.max_model_len:
                drafts.append([])
                continue

            # Register request on first sight; feed the cache each step.
            if state is None:
                index = input_batch.req_id_to_index.get(req_id)
                prompt_tokens: List[int] = []
                if index is not None:
                    try:
                        npt = int(input_batch.num_prompt_tokens[index])
                    except Exception:
                        npt = 0
                    if npt > 0:
                        prompt_tokens = _read_row(
                            input_batch, index, npt
                        )
                    else:
                        # v0.29 adapter shape: prompt length is unknown.
                        # The full row content (prompt + generated so far)
                        # is the best available context, and a frequency
                        # suffix cache tolerates the small overlap.
                        prompt_tokens = _read_row(input_batch, index, num_tokens)
                self._active[req_id] = {"active_tokens": list(prompt_tokens), "prev_draft": []}
                self.suffix_cache.add_prompt(prompt_tokens)
                state = self._active[req_id]
            state["active_tokens"].extend(int(t) for t in sampled_ids)

            max_draft = min(num_spec, self.max_model_len - num_tokens - 1)
            if max_draft <= 0:
                drafts.append([])
                continue

            context = _read_row(input_batch, i, num_tokens)
            suffix_draft, suffix_score, match_len = self.suffix_cache.speculate(
                context, max_draft
            )
            match_lens.append(match_len)

            ngram_draft: List[int] = []
            ngram_score = 0.0
            if self.use_ngram:
                ngram_proposer = self._get_ngram_proposer()
                if ngram_proposer is not None:
                    ngram_draft, ngram_score = self._native_ngram_draft(
                        ngram_proposer, sampled_token_ids, num_tokens_no_spec,
                        input_batch, i, max_draft,
                    )

            if ngram_score > suffix_score and ngram_draft:
                drafts.append(ngram_draft)
                chosen_this_step.append("ngram")
                self._ngram_proposals += 1
            elif suffix_draft:
                drafts.append(suffix_draft)
                chosen_this_step.append("suffix")
                self._suffix_proposals += 1
            else:
                drafts.append([])
            state["prev_draft"] = drafts[-1]

        # Finalize requests that disappeared from the batch: their finished
        # output sequence enters the corpus — this is what makes the cache
        # improve over time.
        active_ids = set()
        for rid in req_ids:
            active_ids.add(rid)
        for rid in [r for r in self._active if r not in active_ids]:
            state = self._active.pop(rid)
            self.suffix_cache.add_sequence(state["active_tokens"])

        self._match_len_sum += sum(match_lens)
        self._match_len_count += len(match_lens)
        self._maybe_emit_stats()
        return drafts

    def _native_ngram_draft(
        self, ngram_proposer: Any, sampled_token_ids: List[List[int]],
        num_tokens_no_spec: Any, input_batch: Any, row: int, max_draft: int,
    ) -> Tuple[List[int], float]:
        """Run vLLM's native in-request n-gram drafter for one row.

        The native NgramProposer.propose signature is
        ``propose(num_speculative_tokens, sampled_token_ids, num_tokens_no_spec,
        token_ids_cpu)`` with numpy arrays. We build minimal numpy inputs for
        the single row (numpy is present in vllm images; if not, we bail to
        suffix-only). Score: match_len-style heuristic — the native proposer
        does not return match info, so we score it as slightly below a
        suffix match of the same length: a constant 0.5 * len(draft) stands
        in for empirical frequency; documented approximation. Because it
        only matches within the request itself, it cannot contribute
        cross-request knowledge — the suffix cache wins whenever it has
        real evidence (score = product of empirical frequencies).
        """
        try:
            import numpy as np
        except Exception:
            return [], 0.0
        try:
            num_tokens = int(num_tokens_no_spec[row])
            context = _read_row(input_batch, row, num_tokens)
            if len(context) < self.ngram_min:
                return [], 0.0
            k = min(max_draft, len(context))
            sampled = [[t] for t in [0]]  # single synthetic new token
            # Build per-row inputs shaped as the native proposer expects.
            cpu = np.zeros((1, max(1, len(context))), dtype=np.int32)
            cpu[0, : len(context)] = context
            nts = np.array([len(context)], dtype=np.int32)
            # sampled_token_ids for the row: last sampled id (the newest token).
            newest = int(sampled_token_ids[row][-1])
            out = ngram_proposer.propose(k, [[newest]], nts, cpu)
            draft = out[0] if out else []
            if not draft:
                return [], 0.0
            return list(draft), 0.5 * min(len(draft), self.ngram_max)
        except Exception:
            return [], 0.0

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _maybe_emit_stats(self) -> None:
        if self._calls % self.stats_interval == 0 and self.stats_file:
            import json

            record = {
                "ts": time.time(),
                "proposed": self._proposed,
                "accepted": self._accepted,
                "suffix_proposals": self._suffix_proposals,
                "ngram_proposals": self._ngram_proposals,
                "avg_match_len": (
                    self._match_len_sum / self._match_len_count
                    if self._match_len_count
                    else 0.0
                ),
            }
            try:
                with open(self.stats_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as exc:
                logger.warning("suffix_hybrid: cannot write stats file: %s", exc)
        if self._calls % self.log_interval == 0:
            acc_rate = (self._accepted / float(self._proposed)) if self._proposed else 0.0
            logger.info(
                "suffix_hybrid: calls=%d proposed=%d accepted=%d (%.1f%%) "
                "suffix=%d ngram=%d",
                self._calls, self._proposed, self._accepted, acc_rate * 100.0,
                self._suffix_proposals, self._ngram_proposals,
            )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "calls": self._calls,
            "proposed": self._proposed,
            "accepted": self._accepted,
            "acceptance_rate": (
                self._accepted / float(self._proposed) if self._proposed else 0.0
            ),
            "suffix_proposals": self._suffix_proposals,
            "ngram_proposals": self._ngram_proposals,
            "avg_match_len": (
                self._match_len_sum / self._match_len_count
                if self._match_len_count
                else 0.0
            ),
            "active_requests": len(self._active),
            "cache": self.suffix_cache.stats(),
        }

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        """vLLM proposer interface: no model to load."""
        pass


class _PatchedConfig:
    """Duck-typed stand-in for VllmConfig with our n-gram min/max.

    The native NgramProposer reads speculative_config.prompt_lookup_min/max
    (zeroed in custom_class mode), scheduler_config.max_num_seqs, and
    model_config.max_model_len. We forward other attribute access to the
    real config when available.
    """

    def __init__(self, base: Any, min_n: int, max_n: int, k: int, max_model_len: int) -> None:
        self._base = base
        self.speculative_config = _SpecPatch(min_n, max_n, k)
        self.model_config = _ModelPatch(max_model_len)
        self.scheduler_config = _SchedulerPatch(base)
        self.parallel_config = _ParallelPatch(base)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_base"), name)


class _SpecPatch:
    def __init__(self, min_n: int, max_n: int, k: int) -> None:
        self.prompt_lookup_min = min_n
        self.prompt_lookup_max = max_n
        self.num_speculative_tokens = k


class _ModelPatch:
    def __init__(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len


class _SchedulerPatch:
    def __init__(self, base: Any) -> None:
        self._base = base

    def __getattr__(self, name: str) -> Any:
        base = object.__getattribute__(self, "_base")
        return getattr(getattr(base, "scheduler_config", None), name,
                       getattr(_Defaults(), name))


class _ParallelPatch:
    def __init__(self, base: Any) -> None:
        self._base = base
        self.tensor_parallel_size = 1

    def __getattr__(self, name: str) -> Any:
        base = object.__getattribute__(self, "_base")
        return getattr(getattr(base, "parallel_config", None), name,
                       getattr(_Defaults(), name))


class _Defaults:
    def __getattr__(self, name: str) -> Any:
        return 1024 if name == "max_num_seqs" else 1


def _read_row(input_batch: Any, index: int, num_tokens: int) -> List[int]:
    """Read token ids for one row from input_batch.token_ids_cpu.

    token_ids_cpu is a 2D torch tensor in real vLLM (call .numpy() then
    slice); tests supply a duck-typed object. Returns a plain list; never
    raises (returns [] on any failure — a missing row just means no draft).
    """
    try:
        cpu = input_batch.token_ids_cpu
        row = cpu[index]
        np_row = getattr(row, "numpy", None)
        if callable(np_row):
            row = np_row()
        tolist = getattr(row, "tolist", None)
        if callable(tolist):
            row = tolist()
        row = list(row)[:num_tokens]
        return [int(t) for t in row]
    except Exception:
        return []


def _shape_len(obj: Any) -> Optional[int]:
    """Tensor rank for torch tensors / numpy arrays / nested lists, else None.

    Used by propose() introspection to tell a 2-D token_ids_cpu apart from
    a 1-D num_tokens_no_spec without importing torch.
    """
    try:
        shape = getattr(obj, "shape", None)
        if shape is not None and len(shape) >= 1:
            return len(shape)
    except Exception:
        pass
    # Nested python lists: probe one level.
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
        return 2
    return None


class _RowAdapter:
    """Duck-typed InputBatch for the vLLM >= 0.29 custom_class call shape.

    v0.29.0's gpu_model_runner calls the custom proposer positionally as
    ``propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu,
    slot_mappings=...)`` — no InputBatch object, hence no request ids and
    no per-row prompt lengths. This adapter synthesizes both:

    - ``req_ids``/``req_id_to_index``: synthetic ids keyed by ROW position
      plus a generation counter. A row is considered the SAME request while
      ``cur_tokens == prev_tokens + len(prev sampled)`` (the exact continue
      invariant of a decode step under spec decode); when a row's token
      count does not extend the previous sequence, the row is treated as a
      NEW request and the old one is finalized into the suffix corpus
      (its full sequence enters the cache — the self-improvement path).
    - ``num_prompt_tokens``: unknown in this shape; reported as 0 so the
      proposer seeds state from the first sampled tokens. The prompt still
      reaches the corpus on finalize because active_tokens grows from what
      IS observable (the sampled ids plus prior row content).
    """

    def __init__(self, token_cpu: Any, num_tokens_no_spec: Any, num_rows: int) -> None:
        self.token_ids_cpu = token_cpu
        self._nts_list = self._to_list(num_tokens_no_spec)
        self._prev: Dict[int, Tuple[str, int, int]] = {}  # row -> (req_id, prev_tokens, prev_sampled_len)
        self._gen = 0
        req_ids: List[str] = []
        self.req_id_to_index: Dict[str, int] = {}
        self.num_prompt_tokens: List[int] = []
        for row in range(num_rows):
            self._gen += 1
            rid = f"row{row}-gen{self._gen}"
            req_ids.append(rid)
            self.req_id_to_index[rid] = row
            self.num_prompt_tokens.append(0)
            cur = self._nts_list[row] if row < len(self._nts_list) else 0
            self._prev[row] = (rid, cur, 0)
        self.req_ids = req_ids

    def update(self, token_cpu: Any, num_tokens_no_spec: Any, num_rows: int) -> None:
        """Swap in this step's tensors (row count can change between steps)."""
        self.token_ids_cpu = token_cpu
        self._nts_list = self._to_list(num_tokens_no_spec)

    @staticmethod
    def _to_list(x: Any) -> List[int]:
        try:
            if x is None:
                return []
            tolist = getattr(x, "tolist", None)
            if callable(tolist):
                x = tolist()
            return [int(v) for v in list(x)]
        except Exception:
            return []

    def refresh(self, sampled_token_ids: List[List[int]]) -> None:
        """Re-key rows whose token count did not extend the previous step.

        A row continues the SAME request iff
        ``cur_tokens == prev_tokens + prev_sampled_len`` — the exact
        invariant of a decode step (num_tokens_no_spec grows by exactly the
        tokens sampled last step, draft-accepted or not). Anything else
        (count reset, row shrunk, batch reshuffled) means the row now holds
        a different request: assign a fresh synthetic id, which the proposer
        treats as a new request and finalizes the old one into the suffix
        corpus — the self-improvement path.
        """
        new_prev: Dict[int, Tuple[str, int, int]] = {}
        req_ids: List[str] = []
        self.req_id_to_index = {}
        self.num_prompt_tokens = []
        for row in range(len(sampled_token_ids)):
            cur = self._nts_list[row] if row < len(self._nts_list) else 0
            prev = self._prev.get(row)
            if prev is not None and cur == prev[1] + max(prev[2], 1):
                rid = prev[0]
            else:
                self._gen += 1
                rid = f"row{row}-gen{self._gen}"
            sampled_len = len(sampled_token_ids[row]) if row < len(sampled_token_ids) else 0
            req_ids.append(rid)
            self.req_id_to_index[rid] = row
            self.num_prompt_tokens.append(0)
            new_prev[row] = (rid, cur, sampled_len)
        self._prev = new_prev
        self.req_ids = req_ids

    @property
    def num_tokens_no_spec(self) -> List[int]:
        return self._nts_list