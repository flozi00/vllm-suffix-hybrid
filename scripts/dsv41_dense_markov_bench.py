#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSpark Markov head chain bench (one GPU, no model). The code lives in
suffix_hybrid/kernels/dsv41_dense_lab.py (stage 1 of the lane-D lab session, which runs it FIRST);
this wrapper is the standalone entry:  python3 scripts/dsv41_dense_markov_bench.py [--rows 1 8 32]

vLLM 39983a5e7 (v1/worker/gpu/spec_decode/dspark/speculator.py `_sample_sequential`, taken because the
checkpoint config has no `dspark_draft_topk`) runs per draft position i < 5
    bias_i = markov_w2(markov_w1[prev_i])     # [B, 256] @ [129280, 256]^T, BF16, REPLICATED
    token_i = sample(base_logits_i + bias_i)  # prev_{i+1} depends on it: the 5 calls are serial
= five 66 MB weight reads per step per rank (331 MB) unless the 128 MB L2 keeps W2 resident.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from suffix_hybrid.kernels.dsv41_dense_lab import markov_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(markov_main())
