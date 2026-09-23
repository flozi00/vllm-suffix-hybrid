# SPDX-License-Identifier: Apache-2.0
"""Wrap-mode first: suffix arbitration around a model-based drafter.

The primary serving shape is MTP/dflash/dspark native drafting with the
Rust mixer arbitrating per-row suffix tails into the native draft lists
(see wrap.py install()). Standalone suffix-only serving (custom_class
with no drafter) is a dev fallback only: it forces the V1 runner, drops
async scheduling, and measured SLOWER than plain decode on a dense
distill (29.4 vs 47.1 tok/s) — never roll it out to production.
"""
import json
import logging
import os
import time

from suffix_hybrid._native import Engine, VERSION
from suffix_hybrid.wrap import install

logger = logging.getLogger("suffix_hybrid")


class HybridProposer:
    def __init__(self, vllm_config=None):
        install()  # Explicit error if unsupported wrap mode was requested.
        spec = getattr(vllm_config, "speculative_config", None)
        model = getattr(vllm_config, "model_config", None)
        self._engine = Engine(getattr(spec, "num_speculative_tokens", 8),
                              getattr(model, "max_model_len", 32768))
        self.suffix_cache = self._engine.suffix_cache
        self.num_speculative_tokens = self._engine.num_speculative_tokens
        self.use_ngram = self._engine.use_ngram
        self.stats_file = os.environ.get("SUFFIX_HYBRID_STATS_FILE", "").strip()
        logger.info("suffix_hybrid version=%s backend=rust acceptance_source=vllm_metrics_only", VERSION)

    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, *, slot_mappings=None):
        started = time.perf_counter_ns()
        drafts = self._engine.propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu)
        elapsed = time.perf_counter_ns() - started
        log_due, stats_due = self._engine.record_total(elapsed)
        if log_due or (stats_due and self.stats_file):
            stats = self.get_stats()
            stats["ts"] = time.time()
            if log_due:
                logger.info("suffix_hybrid %s", json.dumps(stats, sort_keys=True))
            if stats_due and self.stats_file:
                try:
                    with open(self.stats_file, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(stats, sort_keys=True) + "\n")
                except OSError:
                    logger.exception("suffix_hybrid cannot write telemetry")
            self._engine.record_total(time.perf_counter_ns() - started - elapsed)
        return drafts

    def get_stats(self):
        return self._engine.get_stats()
