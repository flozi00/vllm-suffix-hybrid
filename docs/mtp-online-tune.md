# Online idle-time MTP draft-head tuning (`SUFFIX_MTP_TUNE=1`)

Problem (prod GLM-5.3, 2026-09-29): 3.72 accepted tokens/step on German vs
4.50 on English. The checkpoint MTP head was trained mostly on EN/ZH.
This feature adapts the head to live traffic using only idle GPU time.
Drafts are always target-verified, so a worse head costs speed, never
correctness. Promotions are still gated.

Code: `suffix_hybrid/mtp_tune/`. The full design, env knobs, open risks
(R1-R8) and the pod-validation protocol live in the package docstring
(`suffix_hybrid/mtp_tune/__init__.py`). Summary:

| piece | what it does |
|---|---|
| `capture.py` | wraps `speculator.propose` on rank 0. For a crc32-sampled set of requests it copies the drafter's input hidden rows, token ids, positions, `num_rejected` and the served depth-1 draft into device staging, then D2H on a side stream into a pinned FIFO ring (no host sync; steps are dropped when both slots are busy). Rows are joined per request into windows of W anchors + k+1 label rows. |
| `lora.py` | zero-init LoRA on the MTP dense linears through a per-layer `quant_method` proxy, so it works with any quant method. Sharded like the layer (column: B sharded; row: A sharded), so serving needs no extra collective. Updates are in place only, which keeps it CUDA-graph safe. Also holds the Megatron autograd collectives. |
| `train.py` | functional replay of the Qwen4Exp MTP layer over windows. Frozen linears: forward via `quant_method.apply`, dx backward via probe. Frozen FP8 routed experts: per-expert dequantized tiles, dx plus d(topk_w), no dW, no shadow copy. Also dense causal window attention, vocab-parallel CE, and AdamW on the candidate. |
| `gate.py` | replays the greedy k-depth draft chain on held-out windows for live vs candidate. Promotes when mean accepted length gains >= margin over >= min anchors. Keeps per-language buckets and a parity metric. |
| `scheduler.py` | `EngineCoreProc` idle hook: one `collective_rpc` micro-step per poll while `has_work()` is False, after a grace period. All TP ranks run each step in lockstep. Rank 0 plans and broadcasts. Exceptions disable tuning; a liveness vote keeps that uniform across ranks. |

Status: every component above is proven on CPU (`tests/test_mtp_tune.py`),
including TP=2 over gloo matching TP=1 gradients. The replay's fidelity to
the real vLLM kernels, latency, memory and NCCL behaviour are unverified
until the qwen dev-pod protocol runs. GLM-5.3 (`deepseek_v32/nvidia/mtp.py`:
MLA + DSA, NVFP4 experts, `_eh_plan` bypass) is phase 2: its training replay
is not written yet and the feature fails closed there.
