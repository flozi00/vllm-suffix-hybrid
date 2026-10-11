# In-place SFT of the served model (`SUFFIX_SFT=1`)

LoRA fine-tuning of the **running** vLLM model through an HTTP API. There is
one model: zero-init adapters sit on the served linears from boot, training
runs the same live modules (same NVFP4 kernels, same TP shards), and a
finished run is copied into the live adapters in place, so the next serving
step uses it. Checkpoints follow parakeet-server's online-training contract
(private Hugging Face repo, `latest.json`, resumed at startup).

Code: `suffix_hybrid/sft/`. Design, env knobs and risks are in the package
docstring (`suffix_hybrid/sft/__init__.py`). Tests: `tests/test_sft.py` (CPU).

## Enable

| env | value |
|---|---|
| `SUFFIX_SFT` | `1` |
| `VLLM_PLUGINS` | `suffix_sft,<every suffix_* kernel plugin the pod uses>` (VLLM_PLUGINS also filters general plugins) |
| `SUFFIX_SFT_REPO_ID` | private model repo, e.g. `primeline/qwen-fable-sft` (unset = local checkpoints only) |
| `HF_TOKEN` | write token for that repo |

Leave VRAM headroom for training: `(1 - gpu_memory_utilization) * 96 GB`
must cover `SUFFIX_SFT_MIN_FREE_GIB` (default 4) plus 2x the adapter size.
With `--api-key` set, `/v1/sft/*` require it like every `/v1` route. Call the
deployment's own hostname: the unified proxy routes `/v1/*` by `model`, so
`train` / `load` accept an ignored `"model"` field, but `GET status` has no body.

## API

```bash
# train: loss on the final assistant message (or the raw completion)
curl -s $HOST/v1/sft/train -H 'Content-Type: application/json' -d '{
  "examples": [
    {"messages": [{"role": "user", "content": "Wer ist Primeline?"},
                  {"role": "assistant", "content": "Ein deutscher IT-Dienstleister."}]},
    {"prompt": "Q: 2+2=\nA:", "completion": " 4"}
  ],
  "steps": 20, "batch_size": 2, "learning_rate": 1e-4,
  "chat_template_kwargs": {"enable_thinking": false}
}'
# -> {run_id, loss_first, loss, parity_before, parity_after, commit_sha, served: true, ...}

curl -s $HOST/v1/sft/status
curl -s $HOST/v1/sft/load -d '{"run_id": "<run_id>"}'   # serve a published run
curl -s $HOST/v1/sft/load -d '{"run_id": "base"}'       # back to the base model
```

A run trains from the adapter currently served (runs stack). Only one run or
load at a time (409). Every step is one RPC; under load the API yields
`SUFFIX_SFT_DUTY` (default 0.5) of GPU time back to serving, and a request
arriving mid-step waits at most one micro-step (bounded by
`SUFFIX_SFT_MAX_TOKENS`).

Guards per run: the training forward must match serving within
`SUFFIX_SFT_PARITY_NATS` mean NLL on the first example (else 409); every
adapter must receive gradient on the first micro-step (else the run fails:
some op in the forward has no autograd); free VRAM is voted across TP ranks
before every micro-step; a non-finite loss / grad norm aborts; any TP rank
failing fails the op on all. Failed runs never reach serving; a failure while
committing / loading resets every rank to the base model (status says so)
rather than leave TP shards on different adapters. After every commit / load
the prefix cache is reset with running requests preempted, so no KV or GDN
state computed under the old adapter is reused.

## Checkpoints

```
<repo>/latest.json                         -> run metadata, or {"checkpoint": null}
<repo>/runs/<id>/adapter-tp<r>-of-<n>.safetensors
<repo>/runs/<id>/metadata.json             fingerprint, model, rank, alpha, targets, tp, loss, ...
```

Each run is committed atomically (files + `latest.json`, against the parent
commit). At startup the server loads `latest.json`'s run before it accepts
requests; a missing repo or pointer means the base model. If the restore
fails (network, a checkpoint for another model / TP / rank), the server serves
the base model with status `restore_failed` and refuses training until
`/v1/sft/load` picks a checkpoint. `load` also moves `latest.json`, so a
rollback survives restarts.

## Support

Dense attention + Qwen GDN hybrids (Qwen3.5 / qwen-fable), Gemma-4 dense
(sliding + full attention, KV sharing), any TP. Refused with a logged reason,
serving untouched: MoE, MLA, pipeline parallel, soft-capped attention, other
recurrent layers. Vision towers are not trained and keep working.

## Measured on silicon (qwen3.8-27b-fable NVFP4, 1x RTX PRO 6000, v0.30.0, 2026-10-11)

Dev pool `fable-sft-dev` = prod qwen-fable spec (V2 runner, MTP k=2, NVFP4 KV /
GEMM / lm_head) + SFT, `gpu_memory_utilization` 0.80.

| check | result |
|---|---|
| install | 256 adapters (r=16, 97M params); compile + CUDA graphs fine |
| memorization (6 ex, 30 steps, lr 2e-4) | loss 2.71 -> 4e-5 in 142 s; served answers the trained fact, also paraphrased |
| served = trained | after commit served NLL 1.7e-5 == trained 2e-5 |
| parity, <= 1k tokens | mean 0.014-0.035 nats/token (serving vs serving with KV dtype changed: 0.002-0.02) |
| parity, > 2k tokens | serving's own NLL drifts up (1.22 vs 1.10 at 3.4k) -> parity is measured on 1024 tokens |
| load base / run, restart restore | ok; restart serves the Hub's latest.json run before traffic |
| training under 8-client load | 324 requests, 0 errors, p50 1.8 s, max 6.7 s (waits <= one micro-step) |
| micro-step time | 2.5 s @ 366 tok, 8.75 s @ 3.4k tok (fwd 2.6 s, bwd 6.1 s) |
| training VRAM on top of serving | 0.8-1.3 GiB short, 6.5 GiB @ 3.4k tokens (17 GiB free at util 0.80) |
| pinned host | <= SUFFIX_SFT_OFFLOAD_GIB (+1.2 GiB master/AdamW for 97M params) |
| decode cost of attached adapters (B=0) | c1 96 -> 86 tok/s (-10 %), c8 694 -> 653 (-6 %) |
| decode with a trained adapter | lower again when the MTP head disagrees with the tuned model (R4) |

Note: the pool gateway (EPP) needs `"prompt"` in POST bodies; long runs use
`"wait": false` + status polling (300 s route timeout); SFT needs the Python
frontend (`VLLM_USE_RUST_FRONTEND=0`: the Rust frontend has no endpoint plugins).

## Pod validation protocol (before any prod use)

0. Dev pool, bundle with this branch, `SUFFIX_SFT=1`, `VLLM_PLUGINS` as above.
   Expect `[suffix sft] installed: N adapters ...` per rank and the routes
   line. Greedy outputs match the same pool without the gate (B = 0; not
   bitwise: adapters can change Inductor's quant fusions, risk R1).
1. Decode tok/s and TTFT, gate on vs off (adapter GEMMs + lost fusions, R1).
2. Train a tiny memorization set (e.g. 4 examples, 30 steps): loss falls,
   `parity_before.delta` small, `parity_after.served_nll` follows
   `trained_nll`, the served model now answers with the trained text.
3. Peak VRAM during a 4096-token micro-step vs the headroom; no OOM in
   serving under concurrent load (fire requests during training).
4. Restart the pod: it serves the published run before accepting traffic.
5. TP=2 (gemma-4-31b): same as 2; `load base` then `load <run>`.
6. Fault injection: kill the request mid-run (run still publishes), bad HF
   token (502, nothing served), wrong `SUFFIX_SFT_RANK` vs checkpoint
   (`restore_failed`).
