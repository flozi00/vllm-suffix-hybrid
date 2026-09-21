# SPDX-License-Identifier: Apache-2.0
"""Pinned vLLM native-prefix + Rust suffix-tail ABI adapter.

Initial scope: synchronous scheduling, disable_padded_drafter_batch=True,
all-greedy requests. The complete native drafter runs on every call: no native
compute savings are claimed. Rust owns contexts/cache, split policy and feedback.

Source reasoning (d05da62e9): GPUModelRunner.sample_tokens calls the unpadded
drafter AFTER _bookkeeping_sync writes accepted+bonus tokens to token_ids_cpu.
The padded/async call occurs BEFORE that write and is deliberately rejected.
SpecDecodeBaseProposer.prepare_inputs removes rejected positions; its next
first forward rewrites accepted target positions into drafter KV slots before
continuing. No speculative hidden state is retained by this adapter. Qwen3_5
MTP layers are explicitly full_attention, even with a recurrent target; the
target runner's own verification/rollback is untouched. Other hybrid models
and proposer subclasses have not been audited and are rejected.

Ragged CPU lists are supported by _get_draft_token_ids_cpu without padding.
Greedy rejection ignores q; when native q exists we nevertheless retain its
native-prefix entries and use deterministic one-hot q for the suffix tail.
Stochastic batches are rejected, not silently approximated.

Feedback is sampler output length minus recovery/bonus, with scheduler-trimmed
successes censored. Timing passed to Rust is the PREVIOUS drafting adapter wall
time (including native D2H synchronization), not target verification time or
end-to-end latency. Request IDs and previous lengths are hook bookkeeping only.
"""
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time


# SHA256 of the last fully-audited vLLM contracts (d05da62e9). These are
# telemetry + forensic reference only: a serving image that drifts from the
# pin must NOT crash the pod. Drift is logged loudly to stderr (visible in
# pod logs / metrics sidecar) and execution continues when the live
# capability check below passes. Failing closed on any vendor patch turned
# every image rebuild into a CrashLoopBackOff; failing open with identical
# behavior is safe here because drafts are proposals — the target model's
# rejection sampler still verifies every token, so a changed upstream file
# can only cost acceptance, never correctness.
_SOURCE_HASHES = {
    "v1/worker/gpu_model_runner.py": "74d01726dff08704a1d1684f955d86d49e8643972f2003e1ca2c811b2d47bda9",
    "v1/spec_decode/llm_base_proposer.py": "260ac95740b07fca10b5deb8b1e9f24e89577f35dcbc65db979ae5e5b26e2a77",
    "v1/spec_decode/eagle.py": "b2b1f7d15117b43108be368756d9c6e9334d9415a070e46178e18bb7c6476378",
    "v1/spec_decode/dflash.py": "758318d6058caefbd3dbe91feddfba88006f872309f7c3275c6cc412d8c5733f",
    "v1/sample/rejection_sampler.py": "9acaefa7ff3ff02649adf3a2edc4a7281d0089c23566fe5ff140c5f820d88ac4",
    "model_executor/models/qwen3_5_mtp.py": "a7550c4954899b38cecc536633c0fffd04820ed802610fb28cd4d79971b4f1f1",
}

# Capability contract the hook actually needs from the live runner class.
# Whole files drift with every vendor patch; these shapes are what the
# adapter calls, so these are what install() enforces (fail closed).
# input_batch is deliberately NOT here: it is an instance attribute that
# only exists after engine init, while install() patches the class. The
# per-call wrapper touches self.input_batch and fails loudly there if a
# future vLLM ever removes it.
_REQUIRED_RUNNER_ATTRS = ("propose_draft_token_ids",)
_REQUIRED_PROPOSE_PARAMS = ("scheduler_output", "sampled_token_ids",
                            "sampling_metadata", "spec_decode_metadata",
                            "slot_mappings")


def _fingerprint_sources(root):
    """Compare live sources against the audited pin; warn, never raise.

    Returns the list of drifted relative paths (missing or hash-changed).
    Always safe to call: any filesystem error degrades to 'everything
    drifted' rather than an exception, so a half-mounted image still gets
    the capability check instead of a crash.
    """
    drifted = []
    try:
        for relative, expected in _SOURCE_HASHES.items():
            try:
                path = Path(root) / relative
                if (not path.is_file()
                        or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                    drifted.append(relative)
            except OSError:
                drifted.append(relative)
    except Exception:
        return list(_SOURCE_HASHES)
    return drifted


def _verify_sources(root):
    """Legacy name kept for tests/callers: log drift, do not raise."""
    drifted = _fingerprint_sources(root)
    if drifted:
        print("suffix_hybrid source drift vs d05da62e9 (continuing on "
              f"capability check): {json.dumps(sorted(drifted))}",
              file=sys.stderr, flush=True)
    return drifted


def _check_runner_capability(runner):
    """Fail closed when the live runner lacks the hooked shape.

    Raises RuntimeError naming the missing piece. Pure introspection, no
    imports, no I/O — safe to unit-test with SimpleNamespace fakes.
    """
    for attr in _REQUIRED_RUNNER_ATTRS:
        if not callable(getattr(runner, attr, None)) and not hasattr(runner, attr):
            raise RuntimeError(
                f"suffix hybrid unsupported runner contract: missing {attr!r}")
    original = getattr(runner, "propose_draft_token_ids", None)
    if not callable(original):
        raise RuntimeError(
            "suffix hybrid unsupported runner contract: "
            "propose_draft_token_ids is not callable")
    try:
        import inspect
        params = set(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return original
    missing = [p for p in _REQUIRED_PROPOSE_PARAMS if p not in params]
    if missing:
        raise RuntimeError(
            "suffix hybrid unsupported runner contract: "
            f"propose_draft_token_ids missing params {missing}")
    return original


def _wrap_propose(original, mixer_factory):
    @functools.wraps(original)
    def propose(self, scheduler_output, sampled_token_ids, sampling_metadata,
                hidden_states, sample_hidden_states, aux_hidden_states,
                spec_decode_metadata, common_attn_metadata, slot_mappings):
        spec = self.speculative_config
        draft_config = getattr(spec, "draft_model_config", None)
        draft_type = getattr(getattr(draft_config, "hf_config", None), "model_type", None)
        drafter_type = type(self.drafter)
        if (drafter_type.__module__, drafter_type.__name__) not in {
                ("vllm.v1.spec_decode.eagle", "EagleProposer"),
                ("vllm.v1.spec_decode.dflash", "DFlashProposer")}:
            raise RuntimeError("suffix hybrid unsupported drafter class/state contract")
        if (self.use_async_scheduling or not spec.disable_padded_drafter_batch
                or not sampling_metadata.all_greedy
                or not isinstance(sampled_token_ids, list)
                or self.parallel_config.pipeline_parallel_size != 1
                or (self.model_config.is_hybrid and draft_type != "qwen3_5_mtp")
                or self.rejection_sampler.synthetic_mode
                or spec.method not in {"eagle", "eagle3", "mtp", "dflash", "dspark"}):
            raise RuntimeError("suffix hybrid requires synchronous, unpadded, greedy-only "
                               "Eagle/MTP/DFlash/DSpark; no PP, unaudited recurrent model, or synthetic acceptance")
        started = time.perf_counter_ns()
        native = original(self, scheduler_output, sampled_token_ids, sampling_metadata,
                          hidden_states, sample_hidden_states, aux_hidden_states,
                          spec_decode_metadata, common_attn_metadata, slot_mappings)
        batch = self.input_batch
        ids = list(batch.req_ids)
        drafts = ([list(row) for row in native] if isinstance(native, list)
                  else native.detach().cpu().tolist())
        if not hasattr(self, "_suffix_hybrid_mixer"):
            self._suffix_hybrid_mixer = mixer_factory(self.num_spec_tokens, self.max_model_len)
        for i, sampled in enumerate(sampled_token_ids):
            if not sampled:
                drafts[i] = []
        # parse_output has removed -1 padding; the final token is recovery/bonus,
        # never an accepted draft token. No token-equality acceptance inference.
        verified = ([int(n) for n in spec_decode_metadata.num_draft_tokens]
                    if spec_decode_metadata is not None else [0] * len(ids))
        accepted = [len(sampled) - 1 if sampled and verified[i] else -1
                    for i, sampled in enumerate(sampled_token_ids)]
        previous_lengths = getattr(self, "_suffix_hybrid_lengths", {})
        for i, req_id in enumerate(ids):
            if accepted[i] == verified[i] and verified[i] != previous_lengths.get(req_id):
                # All scheduled tokens accepted, but unseen tail is censored.
                accepted[i] = -1
        mixed = self._suffix_hybrid_mixer.mix_numpy(
            ids, batch.num_tokens_no_spec, batch.token_ids_cpu, drafts, accepted,
            getattr(self, "_suffix_hybrid_cost_ns", None))
        if self._draft_probs is not None:
            import torch
            prefixes = self._suffix_hybrid_mixer.last_native_counts()
            probs = self._draft_probs.clone()
            for i, (row, prefix) in enumerate(zip(mixed, prefixes)):
                if prefix < len(row):
                    tail = probs[i, prefix:len(row)]
                    tail.zero_()
                    indices = torch.tensor(row[prefix:], device=probs.device,
                                           dtype=torch.long).unsqueeze(1)
                    tail.scatter_(1, indices, 1.0)
            self._draft_probs = probs
        self._suffix_hybrid_lengths = dict(zip(ids, map(len, mixed)))
        self._suffix_hybrid_cost_ns = time.perf_counter_ns() - started
        self._suffix_hybrid_calls = getattr(self, "_suffix_hybrid_calls", 0) + 1
        interval = int(os.environ.get("SUFFIX_HYBRID_LOG_INTERVAL", "0"))
        if interval > 0 and self._suffix_hybrid_calls % interval == 0:
            print("suffix_hybrid_native " + json.dumps(self._suffix_hybrid_mixer.get_stats(),
                                                      sort_keys=True), file=sys.stderr, flush=True)
        return mixed
    return propose


def install():
    if os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() != "1":
        return False
    try:
        vllm = importlib.import_module("vllm")
        module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
        from suffix_hybrid._native import HybridMixer
    except (ImportError, OSError) as exc:
        raise RuntimeError("suffix hybrid unsupported installation; requires vLLM and Rust HybridMixer") from exc
    # Drift from the audited pin is telemetry (stderr), never fatal: the
    # capability check below is the real gate.
    _verify_sources(Path(vllm.__file__).parent)
    original = _check_runner_capability(module.GPUModelRunner)
    if not getattr(original, "_suffix_hybrid_hook", False):
        wrapped = _wrap_propose(original, HybridMixer)
        wrapped._suffix_hybrid_hook = True
        module.GPUModelRunner.propose_draft_token_ids = wrapped
    return True
