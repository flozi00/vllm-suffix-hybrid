# Does SFT on generated agent traces actually improve a model?

A reproducible before/after benchmark for the in-place SFT endpoint. The point
is to answer one question with evidence: **does training on our generated
data make the model better, or are we building an imaginary castle?**

Reference run: LFM2.5-2.6B, vLLM 0.32.0, one 12 GB GPU, 839 traceforge
examples, LoRA r=16 (23.0 M params).

## Why two signals

A capability benchmark alone can be gamed by format learning: a model that
learns to emit `{"name": ...}` JSON passes the tool-call check without
learning anything about the distribution. A held-out NLL alone can improve
while the model gets no more useful.

So every run records both:

| signal | what it measures | why it is here |
|---|---|---|
| capability tasks | 90 EN+DE prompts with a mechanically checkable answer | what a user sees |
| held-out NLL | mean nats/token on 94 traces the model never trained on | sensitive to distribution shift on every token |

The verdict rule is deliberately strict: **capability up AND held-out NLL
down** is the only combination that counts as "the data helps". Capability up
with flat NLL means a format was learned, not a distribution.

## Capability tasks

90 items, 45 English / 45 German, five check types, 18 per type:

| check | prompt | pass condition |
|---|---|---|
| `tool_call` | task + tool schema, "reply with one JSON object" | parses as JSON, correct tool name, all required args present |
| `json_keys` | "return a JSON object with keys a, b, c" | parses, all keys present |
| `lang` | question in EN or DE | answer's function-word profile matches the prompt language |
| `bullets` | "give exactly N bullet points" | exactly N bullet lines |
| `no_echo` | a long review brief | answer is not a near-copy of the brief (difflib ratio < 0.6) |

`no_echo` exists because the traceforge pipeline rejects brief-echoing
answers mechanically; if the training data teaches that, it should show up
here.

Language detection is a disjoint function-word count plus an umlaut bonus, not
a model call — the benchmark has no LLM judge and no network dependency.

## Held-out split

Every 10th trace by `crc32(session_id) % 10 == 0` is held out before training
(94 of 933 usable traces). The split is deterministic, so the same traces are
scored before and after.

NLL is measured through `/v1/completions` with `echo=True` and
`prompt_logprobs=1`, scoring only the completion span. The prompt token count
comes from a separate call on the prompt alone — vLLM returns logprobs aligned
to tokens but no character offsets, so a character-fraction guess would
misplace the boundary.

## Training configuration

| knob | value |
|---|---|
| examples | 855 (950 usable minus 95 held out) |
| chunk size | 855 (one request; the endpoint caps at `SUFFIX_SFT_MAX_EXAMPLES`) |
| batch size | 64 — this is gradient accumulation: one optimizer step per 64 examples, split into micro-batches by the token budget |
| steps | 1 epoch = 13 optimizer steps |
| learning rate | 3e-5 |
| max tokens/example | 8192 (micro-batch budget; the model context is 128k) |
| truncation | keep system + first user message, drop the middle, keep the tail |

Truncation keeps the task head: dropping the head instead would train the
model to answer a question it never saw.

The 8192 budget is a memory decision, not a modelling one — see
`sft-deployment.md` for the measured cost curve (3 s/example at 2.3k tokens,
21 s at 5.6k, 405 s at 24k). The corpus median is 24.5k tokens, so full-length
training is a multi-day job on a 12 GB card.

## Reproducing

```bash
# 1. serve
bash run.sh

# 2. build examples + held-out split (inside the container, for the tokenizer)
docker exec lfm-sft python3 /work/sft/build_examples.py \
    --dataset /data/train.jsonl --out /work/sft/examples.jsonl \
    --heldout-out /work/bench/heldout.jsonl --heldout-mod 10

# 3. baseline
python bench/run_bench.py --label base

# 4. train
python sft/train.py --examples sft/examples.jsonl --chunk 200 --epochs 1

# 5. replay every checkpoint and benchmark it (builds the trajectory)
python sft/replay.py --train results/train.json

# 6. compare
python bench/compare.py results/base.json results/after-839.json
python bench/history.py
```

Every benchmark run appends one line to `results/history.jsonl`, so the
trajectory survives across runs and can be rendered with `bench/history.py`.

## Results

<!-- RESULTS:BEGIN -->
LFM2.5-2.6B, 12 GB GPU, 839 traceforge examples, LoRA r=16 (23.0 M params),
1 epoch at batch 64 / lr 3e-5 (13 optimizer steps, 44 min). Base and trained
were benchmarked at the same 128k serving config; only the adapter changed.

| metric | base | after SFT | delta | ±95% | verdict |
|---|---|---|---|---|---|
| overall | 24.4% | 55.6% | +31.1 | 8.9 | better |
| en | 28.9% | 55.6% | +26.7 | 13.2 | better |
| de | 20.0% | 55.6% | +35.6 | 11.7 | better |
| json_keys | 5.0% | 100.0% | +95.0 | 9.6 | better |
| lang | 50.0% | 100.0% | +50.0 | 21.9 | better |
| no_echo | 70.0% | 100.0% | +30.0 | 28.4 | better |
| bullets | 20.0% | 0.0% | −20.0 | 17.5 | worse |
| tool_call | 0.0% | 0.0% | 0.0 | 0.0 | flat |
| held-out NLL | 2.2030 | 1.9864 | −0.2166 (−9.8%) | | better |

Verdict: **the data helps** — capability up and held-out NLL down on 94 traces
the model never trained on. Full write-up: `lfm-sft-proof/RESULTS.md`.
<!-- RESULTS:END -->

## Learnings

<!-- LEARNINGS:BEGIN -->
- **The training forward matches serving.** The endpoint's parity check
  (served NLL vs training-forward NLL on the same example) reported 0.0087 nats
  before and 0.0023 after the run, against a 0.15-nat tolerance. A wrong
  attention mask or loss scale would not stay inside that.
- **`SUFFIX_SFT_MAX_TOKENS` is the micro-batch budget, not the context.**
  `--max-model-len` is the context. Conflating them makes a memory limit look
  like a context limit.
- **Long-context training is memory-bound, not flag-bound.** 128k inference
  fits at 8.3 GiB; a 24k-token training example costs 405 s because the worker
  checkpoints and host-offloads. Budget ~3 s/example at 2.3k tokens, 21 s at
  5.6k, 405 s at 24k.
- **Raising the micro-batch budget can be slower than lowering it.** At 8k the
  training peak collided with the 128k KV pool, leaving ~200 MiB free; the
  allocator thrashed and one optimizer step took two hours instead of twenty
  minutes.
- **A rejected request still returns 202.** The worker drops the payload and
  records the reason in `status.rejected`; a driver that only polls `state`
  reports success with `loss None`. Treat a missing `last_run` as a failure.
- **Leave headroom under the token cap.** Examples built to exactly 8192 tokens
  were rejected at 8195 — the plugin re-renders and adds special tokens.
- **Capability saturates before NLL does.** lr 1e-4 / batch 1 / 200 steps and
  lr 3e-5 / batch 64 / 13 steps both reach 55.6% capability, but the second is
  0.076 nats better on held-out NLL.
- **A metric can regress for a data reason.** `bullets` went 20% → 0%: the
  corpus is coding traces whose answers are prose and code, so the adapter
  unlearned the base model's list habit. Coverage gaps show up here first.
<!-- LEARNINGS:END -->
