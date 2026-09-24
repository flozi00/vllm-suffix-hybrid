# SPDX-License-Identifier: Apache-2.0
"""K1 spike: correctness of the fused greedy spec-decode verify op.

Reference = a faithful torch re-implementation of vLLM 0.30.0's greedy
rejection path (vllm/v1/sample/rejection_sampler.py):
  * `target_logits.argmax(dim=-1)`                (:468)
  * per-request accept-run until first mismatch  (:748-761)
  * bonus slot after a full accept                (:767-773)

The Rust op `suffix_hybrid._native.rejection_greedy_accept` fuses the same
computation (draft-token compare + accept-mask) in one host pass; the GPU
(cutile) variant is the entry of the K1 kernel and must match this oracle
bit-for-bit on int outputs.

Runs CPU-side (torch + numpy only). Run: pytest tests/test_verify_fusion.py
"""
import unittest

import numpy as np
import torch

from suffix_hybrid._native import rejection_greedy_accept

K = 8  # dev-pool draft length


def torch_reference(draft_token_ids, target_logits, cu, max_spec_len):
    """vLLM greedy verify semantics, in torch, on CPU.

    draft_token_ids: [num_tokens] i64 (flattened, -1 = padding)
    target_logits:   [num_tokens, vocab] f32 -- argmax per row
    cu:              [batch] i64 exclusive cumsum of num_draft_tokens
    returns (output_token_ids [batch, k+1] i32 with -1 placeholders,
             accept_count [batch] i32)
    """
    num_tokens = draft_token_ids.shape[0]
    batch = cu.shape[0]
    target_argmax = target_logits.to(torch.float32).argmax(dim=-1)
    out = torch.full((batch, max_spec_len + 1), -1, dtype=torch.int32)
    counts = torch.zeros(batch, dtype=torch.int32)
    for b in range(batch):
        start = 0 if b == 0 else int(cu[b - 1])
        end = min(int(cu[b]), num_tokens)
        emitted = 0
        acc = 0
        rejected = False
        for p in range(start, end):
            d = int(draft_token_ids[p])
            if d >= 0 and d == int(target_argmax[p]):
                out[b, emitted] = d
                emitted += 1
                acc += 1
            else:
                # Mismatch OR padded (-1) draft: vLLM's kernel stores the
                # target argmax here and stops (rejection_sampler.py:761).
                out[b, emitted] = int(target_argmax[p])
                emitted += 1
                rejected = True
                break
        if not rejected:
            # All drafts accepted: bonus token at slot `acc` -- argmax of the
            # bonus row (the greedy bonus sampler output, :767-773).
            if start + acc < target_argmax.shape[0]:
                out[b, emitted] = int(target_argmax[start + acc])
                emitted += 1
        counts[b] = acc
    return out, counts


def emit_tokens_from_rust(count, mask, draft_token_ids, target_argmax, cu,
                          max_spec_len):
    """Reconstruct vLLM output_token_ids from the fused op's (count, mask).

    Where mask is 1 we emit: draft token while position < count, then the
    target argmax (corrective or bonus). This is the exact token math of
    rejection_greedy_sample_kernel (:760-773).
    """
    draft = torch.as_tensor(draft_token_ids)
    argmax = torch.as_tensor(target_argmax)
    batch = mask.shape[0]
    out = torch.full((batch, max_spec_len + 1), -1, dtype=torch.int32)
    for b in range(batch):
        start = 0 if b == 0 else int(cu[b - 1])
        cnt = int(count[b])
        emitted = 0
        for p in range(cnt):
            out[b, emitted] = int(draft[start + p])
            emitted += 1
        if mask[b, cnt] == 1:
            out[b, emitted] = int(argmax[start + cnt])
    return out


class VerifyFusionTests(unittest.TestCase):
    def _run_case(self, num_drafts, vocab=101, seed=0, max_spec_len=K):
        g = torch.Generator().manual_seed(seed)
        batch = len(num_drafts)
        # Draft tokens: random, with random -1 padding beyond each request's
        # real draft count (the scheduler pads to max_spec_len in vLLM 0.30).
        # vLLM target-model logits: [num_tokens = batch*(1+k), vocab] --
        # every request contributes 1+k rows (one per draft slot + one
        # bonus) even when its real draft count is smaller (padded -1).
        target_logits = torch.randn(
            (batch * (max_spec_len + 1), vocab),
            generator=g, dtype=torch.float32)
        rows = []
        for n in num_drafts:
            draft = torch.randint(0, vocab, (max_spec_len,), generator=g)
            if n < max_spec_len:
                draft[n:] = -1
            rows.append(draft)
        draft_token_ids = torch.cat(rows)
        cu = torch.tensor(np.cumsum([max_spec_len] * batch),
                          dtype=torch.int64)  # inclusive cumsum, per kernel :739-745
        ref_out, ref_counts = torch_reference(
            draft_token_ids, target_logits[:draft_token_ids.shape[0], :], cu,
            max_spec_len)
        argmax_npy = target_logits[:draft_token_ids.shape[0], :].argmax(
            dim=-1).numpy().astype(np.int64)
        count, mask = rejection_greedy_accept(
            draft_token_ids.numpy().astype(np.int64), argmax_npy,
            cu.numpy().astype(np.int64), max_spec_len)
        got_out = emit_tokens_from_rust(
            count, mask, draft_token_ids.numpy(),
            argmax_npy, cu.numpy().astype(np.int64), max_spec_len)
        return ref_out, ref_counts, count, mask, got_out

    def test_exact_match_random_batches(self):
        # Deterministic random sweep: deliverable "bit-consistent accept-mask
        # vs torch reference" (K1 gate).
        rng = np.random.default_rng(1234)
        for trial in range(200):
            batch = int(rng.integers(1, 9))
            num_drafts = [int(rng.integers(0, K + 1)) for _ in range(batch)]
            seed = int(rng.integers(0, 10_000))
            ref_out, ref_counts, count, mask, got_out = self._run_case(
                num_drafts, seed=seed)
            self.assertTrue(
                torch.equal(ref_counts, torch.as_tensor(count)),
                f"trial {trial}: counts differ\n{ref_counts}\n{count}")
            self.assertTrue(
                torch.equal(ref_out, got_out),
                f"trial {trial}: reconstructed outputs differ"
                f"\n{ref_out}\n{got_out}")

    def test_edge_all_accept_and_all_reject(self):
        # All accepted (draft == argmax everywhere) and immediate rejection
        # (draft[0] != argmax[0]) for a single request.
        batch_denom = 1
        vocab = 50
        logits = torch.zeros((K + 1, vocab))
        logits[:, 7] = 3.0  # argmax is always 7
        drafts_ok = torch.full((K,), 7, dtype=torch.int64)
        drafts_bad = torch.full((K,), 9, dtype=torch.int64)
        cu = torch.tensor([K], dtype=torch.int64)
        ref_out_ok, ref_count_ok = torch_reference(
            drafts_ok, logits[:K], cu, K)
        count, mask = rejection_greedy_accept(
            drafts_ok.numpy(), logits[:K].argmax(-1).numpy(), cu.numpy(), K)
        self.assertEqual(int(ref_count_ok[0]), int(count[0]))
        self.assertEqual(int(count[0]), K)  # all accepted
        self.assertEqual(mask.sum(), K + 1)  # drafts + bonus slot
        ref_out_bad, ref_count_bad = torch_reference(
            drafts_bad, logits[:K], cu, K)
        count_bad, mask_bad = rejection_greedy_accept(
            drafts_bad.numpy(), logits[:K].argmax(-1).numpy(), cu.numpy(), K)
        self.assertEqual(int(ref_count_bad[0]), int(count_bad[0]))
        self.assertEqual(int(count_bad[0]), 0)
        self.assertEqual(int(mask_bad.sum()), 1)  # corrective token only

    def test_padded_drafts_stop_accept(self):
        # num_draft_tokens < K: padding (-1) after real drafts; accept run
        # must stop at the pad even if later tokens would match.
        logits = torch.zeros((K, 40))
        logits[:, 3] = 1.0
        argmax = logits.argmax(-1).numpy()
        drafts = np.full(K, 3, dtype=np.int64)
        drafts[2:] = -1  # only two real drafts
        cu = np.array([K], dtype=np.int64)
        count, mask = rejection_greedy_accept(drafts, argmax, cu, K)
        self.assertEqual(int(count[0]), 2)
        self.assertEqual(int(mask[0].sum()), 3)  # 2 accepted + bonus


if __name__ == "__main__":
    unittest.main()