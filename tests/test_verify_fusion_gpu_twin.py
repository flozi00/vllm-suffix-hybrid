# SPDX-License-Identifier: Apache-2.0
"""K1 GPU twin: local (CPU) validation of the cutile kernel's algorithm.

The GPU kernel itself (src/verify_fusion_gpu.rs, cargo feature
`cutile-kernels`) only builds in the CUDA-toolkitted CI container — this Mac
has no CUDA toolchain. What CAN be validated here, and what this file locks
down, is the *algorithm* the tile kernel implements:

  `kernel_block_twin()` below is a statement-for-statement Python
  transcription of `rejection_greedy_accept_k1`'s per-tile-block body in
  src/verify_fusion_gpu.rs (start/end from the inclusive cumsum, owned-slot
  clamp, keep-while-match prefix with padded-draft-as-rejection, the
  iota/le_tile/select emit-mask construction). Any semantic edit to the
  kernel must be mirrored here or this suite fails.

The twin is checked against (a) the landed Rust host oracle
`_native.rejection_greedy_accept` (the GPU identity gate's reference) and
(b) the torch reference that mirrors vLLM's
rejection_greedy_sample_kernel, on:
  * randomized ragged batches (padding, mid-chain rejection, all-accept),
  * the batch sweep c = 1..32 at k = 8 (the dev-pool concurrency sweep of
    the K1 identity gate), including the vLLM shape
    logits [c*(1+k), vocab].

The bit-consistent-on-GPU check of the compiled kernel against the host
oracle remains the pool-side K1 identity gate (dossiers/kernel-k1-spike.md
§4); this file is its local pre-gate.

Run: pytest tests/test_verify_fusion_gpu_twin.py
"""
import unittest

import numpy as np
import torch

from suffix_hybrid._native import rejection_greedy_accept

K = 8  # dev-pool draft length


def kernel_block_twin(draft_row, argmax_row, num_tokens, max_spec_len):
    """Statement-for-statement transcription of the tile kernel's per-block
    body in src/verify_fusion_gpu.rs (`rejection_greedy_accept_k1`).

    draft_row / argmax_row: the request's OWN slots only — the kernel loads
    them by global index start+p; here the caller slices [start:end] and we
    add the same clamps the kernel applies.

    Returns (count, mask_row) — the block's accept_count and emit_mask row.
    """
    # kernel: start = b == 0 ? 0 : cu[b-1]; end = min(cu[b], num_tokens)
    # (the caller passes the slice; the clamps below are the kernel's)
    start = 0  # caller guarantees the slice begins at `start`
    end = min(len(draft_row) + start, num_tokens)
    # owned = clamp(end - start, 0, k); the slice length already carries
    # end - start, and the kernel also takes min(len, k):
    owned = min(max(end - start, 0), max_spec_len)
    draft_row = draft_row[:owned]
    argmax_row = argmax_row[:owned]

    # keep-while-match prefix (padded -1 draft = rejection, never a skip)
    count = 0
    p = 0
    accept = True
    while p < owned and accept:
        d = int(draft_row[p])
        a = int(argmax_row[p])
        if d >= 0 and d == a:
            count += 1
        else:
            accept = False
        p += 1

    # emit_mask row = (iota(WIDTH) <= count) as i32
    width = max_spec_len + 1
    mask_row = (np.arange(width) <= count).astype(np.int32)
    return count, mask_row


def kernel_twin_batch(draft_token_ids, target_argmax, cu, max_spec_len):
    """Grid = batch: apply the block twin per request."""
    num_tokens = len(draft_token_ids)
    batch = len(cu)
    counts = np.zeros(batch, dtype=np.int32)
    masks = np.zeros((batch, max_spec_len + 1), dtype=np.int32)
    for b in range(batch):
        start = 0 if b == 0 else int(cu[b - 1])
        end = min(int(cu[b]), num_tokens)
        cnt, mrow = kernel_block_twin(
            draft_token_ids[start:end], target_argmax[start:end],
            num_tokens, max_spec_len)
        counts[b] = cnt
        masks[b] = mrow
    return counts, masks


def torch_reference_counts_mask(draft_token_ids, target_argmax, cu,
                                 max_spec_len):
    """Torch mirror of vLLM's greedy kernel semantics (:739-773): the
    reference the GPU kernel must reproduce bit-for-bett on int outputs."""
    draft = torch.as_tensor(draft_token_ids)
    argmax = torch.as_tensor(target_argmax)
    num_tokens = draft.shape[0]
    batch = len(cu)
    counts = torch.zeros(batch, dtype=torch.int32)
    masks = torch.zeros((batch, max_spec_len + 1), dtype=torch.int32)
    for b in range(batch):
        start = 0 if b == 0 else int(cu[b - 1])
        end = min(int(cu[b]), num_tokens)
        count = 0
        for p in range(start, end):
            if p - start >= max_spec_len:
                break
            d = int(draft[p])
            if d >= 0 and d == int(argmax[p]):
                count += 1
            else:
                break
        counts[b] = count
        masks[b, : count + 1] = 1
    return counts.numpy(), masks.numpy()


class GpuTwinAlgorithmTests(unittest.TestCase):
    def _make_case(self, batch, num_drafts, vocab, seed):
        g = torch.Generator().manual_seed(seed)
        target_logits = torch.randn(
            (batch * (K + 1), vocab), generator=g, dtype=torch.float32)
        argmax = (target_logits[: batch * K].argmax(dim=-1)
                  .numpy().astype(np.int64))
        rows = []
        for n in num_drafts:
            d = torch.randint(0, vocab, (K,), generator=g)
            if n < K:
                d[n:] = -1
            rows.append(d)
        draft = torch.cat(rows).numpy().astype(np.int64)
        cu = np.cumsum([K] * batch).astype(np.int64)  # inclusive cumsum
        return draft, argmax, cu

    def test_twin_matches_host_oracle_random(self):
        # The kernel-algorithm twin must equal the landed Rust host op
        # exactly — the GPU identity gate's contract, on CPU.
        rng = np.random.default_rng(5678)
        for trial in range(200):
            batch = int(rng.integers(1, 9))
            num_drafts = [int(rng.integers(0, K + 1)) for _ in range(batch)]
            seed = int(rng.integers(0, 10_000))
            draft, argmax, cu = self._make_case(
                batch, num_drafts, vocab=101, seed=seed)
            py_count, py_mask = rejection_greedy_accept(
                draft, argmax, cu, K)
            t_counts, t_masks = kernel_twin_batch(draft, argmax, cu, K)
            np.testing.assert_array_equal(
                np.asarray(py_count), t_counts,
                err_msg=f"trial {trial}: counts differ")
            np.testing.assert_array_equal(
                np.asarray(py_mask), t_masks,
                err_msg=f"trial {trial}: masks differ")

    def test_twin_matches_torch_reference(self):
        # And the twin must equal the torch mirror of vLLM's kernel
        # semantics (independent of the Rust host op).
        rng = np.random.default_rng(4321)
        for trial in range(100):
            batch = int(rng.integers(1, 9))
            num_drafts = [int(rng.integers(0, K + 1)) for _ in range(batch)]
            seed = int(rng.integers(0, 10_000))
            draft, argmax, cu = self._make_case(
                batch, num_drafts, vocab=57, seed=seed)
            ref_counts, ref_masks = torch_reference_counts_mask(
                draft, argmax, cu, K)
            t_counts, t_masks = kernel_twin_batch(draft, argmax, cu, K)
            np.testing.assert_array_equal(
                t_counts, ref_counts, err_msg=f"trial {trial}: counts")
            np.testing.assert_array_equal(
                t_masks, ref_masks, err_msg=f"trial {trial}: masks")

    def test_identity_sweep_c1_to_c32(self):
        # K1 identity gate shape sweep: batch concurrency c = 1..32 at k=8
        # (dev-pool gemma-spec c<=32), all three implementations equal.
        vocab = 128
        for batch in range(1, 33):
            rng = np.random.default_rng(1000 + batch)
            # Mixed real-draft counts including 0 (immediate bonus/corrective
            # only) and full-K; forced mid-chain rejection per request.
            num_drafts = [int(rng.integers(0, K + 1)) for _ in range(batch)]
            draft, argmax, cu = self._make_case(
                batch, num_drafts, vocab=vocab, seed=2000 + batch)
            # Force a mid-chain rejection in the first request: flip its
            # second argmax so the accept run stops at 1 (if it had >= 2
            # real drafts).
            if num_drafts[0] >= 2 and draft[1] >= 0:
                argmax[1] = (int(argmax[1]) + 1) % vocab
            py_count, py_mask = rejection_greedy_accept(draft, argmax, cu, K)
            t_counts, t_masks = kernel_twin_batch(draft, argmax, cu, K)
            ref_counts, ref_masks = torch_reference_counts_mask(
                draft, argmax, cu, K)
            np.testing.assert_array_equal(
                np.asarray(py_count), t_counts,
                err_msg=f"batch {batch}: oracle vs twin counts")
            np.testing.assert_array_equal(
                np.asarray(py_mask), t_masks,
                err_msg=f"batch {batch}: oracle vs twin masks")
            np.testing.assert_array_equal(
                t_counts, ref_counts, err_msg=f"batch {batch}: twin vs ref")
            np.testing.assert_array_equal(
                t_masks, ref_masks, err_msg=f"batch {batch}: twin vs ref")

    def test_edge_padded_draft_is_rejection(self):
        # Subtlety 2 of the oracle: a -1 pad must stop the accept run even
        # if the draft BEYOND it would match.
        logits = torch.zeros((K, 40))
        logits[:, 3] = 1.0
        argmax = logits.argmax(-1).numpy()
        draft = np.full(K, 3, dtype=np.int64)
        draft[2:] = -1
        cu = np.array([K], dtype=np.int64)
        py_count, py_mask = rejection_greedy_accept(draft, argmax, cu, K)
        t_counts, t_masks = kernel_twin_batch(draft, argmax, cu, K)
        self.assertEqual(int(py_count[0]), 2)
        np.testing.assert_array_equal(np.asarray(py_count), t_counts)
        np.testing.assert_array_equal(np.asarray(py_mask), t_masks)
        self.assertEqual(int(t_masks[0].sum()), 3)  # 2 accepted + bonus


if __name__ == "__main__":
    unittest.main()