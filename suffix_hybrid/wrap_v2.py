# SPDX-License-Identifier: Apache-2.0
"""Opt-in, source-pinned V2 standard-MTP dense-row prototype.

Native propose ALWAYS runs first on every TP rank. d05da62e9 model_runner
2120 postprocesses GPU history before calling propose (2148), then persists
our full-width output (2163) and sends it to the scheduler (2172). The next
AR prefill recomputes verified positions from target history/hidden states
(autoregressive/speculator 245-281, 311-328, 725-753, 859-870); no native
forward or KV repair is skipped. Never stitch a stale native tail onto suffix.

Synchronous owner-only D2H history transfer is intentional prototype overhead,
not a speed claim. Target temperatures are unrestricted: standard rejection
supports deterministic proposals with draft_logits=None. No vocabulary-sized
probabilities are allocated. CUDA/TP correctness and throughput remain live
benchmark gates; this module is NOT connected to sitecustomize.
"""
import functools

import torch


def _wrap_propose(runner, original, mixer, group):
    previous_widths = {}
    @functools.wraps(original)
    def propose(input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync=None,
                dummy_run=False, skip_attn_for_dummy_run=False, mm_inputs=None,
                is_profile=False):
        native = original(input_batch, attn_metadata, slot_mappings,
                          last_hidden_states, aux_hidden_states, num_sampled,
                          num_rejected, last_sampled, next_prefill_tokens,
                          temperature, seeds, dp_sync=dp_sync, dummy_run=dummy_run,
                          skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                          mm_inputs=mm_inputs, is_profile=is_profile)
        output = native.clone()
        if group.rank_in_group == 0:
            idx = input_batch.idx_mapping.long()
            counts = runner.req_states.total_len.gpu[idx].detach().cpu()
            width = int(counts.max()) if counts.numel() else 0
            # Blocking .cpu() completes the postprocess producer stream before
            # numpy reads; never use optimistic CPU mirrors or async output.
            history = runner.req_states.all_token_ids.gpu[:, :width][idx].detach().cpu()
            ids = list(input_batch.req_ids)
            sampled = num_sampled.detach().cpu().tolist()
            rejected = num_rejected.detach().cpu().tolist()
            accepted = []
            for req_id, ns, nr in zip(ids, sampled, rejected):
                a, verified = ns - 1, ns + nr - 1
                previous = previous_widths.get(req_id, 0)
                valid = (ns > 0 and nr >= 0 and 0 < verified <= previous
                         and 0 <= a <= verified
                         and (a < verified or verified == previous))
                accepted.append(a if valid else -1)
            mixed = mixer.mix_numpy(ids, counts.numpy(),
                                    history.numpy(), native.detach().cpu().tolist(),
                                    accepted)
            output.copy_(torch.tensor(mixed, dtype=native.dtype, device=native.device))
            previous_widths.clear()
            previous_widths.update(zip(ids, map(len, mixed)))
        return group.broadcast(output, src=0)
    return propose
