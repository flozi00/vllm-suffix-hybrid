# SPDX-License-Identifier: Apache-2.0
"""Online, idle-time fine-tuning of the MTP speculative-decoding draft head.

Gate: SUFFIX_MTP_TUNE=1 (default OFF; unset = nothing is imported or
patched). Prototype target: qwen3.8-flash-next (Qwen4ExpMTP, vLLM 0.30.0
V2 runner, MTPSpeculator). GLM-5.3 (GlmMoeDsaForCausalLM ->
vllm/models/deepseek_v32/nvidia/mtp.py, MLA + DSA indexer) is phase 2: the
capture / adapters / scheduler / gate are family-neutral, only the training
replay (train.build_functional) is missing and fails closed.

Why: prod GLM-5.3 accepts 3.72 tok/step on German vs 4.50 on English
(2026-09-29): the checkpoint MTP head was trained mostly on EN/ZH. Drafts
are always target-verified, so a worse head costs speed, never
correctness; promotions are still gated.

Architecture (one worker process per TP rank + the EngineCore process)
------------------------------------------------------------------------
  capture.py   speculator.propose wrapper (rank 0): per step, for a
               deterministic crc32(req_id) sample of requests, index_select
               the drafter's input hidden rows / token ids / positions /
               num_rejected into device staging on the main stream, D2H on a
               side stream straight into a pinned ring ARENA, event; next
               steps absorb with event.query() (zero host syncs, dropped
               when both slots are busy). Rows join per request by position
               continuity into windows of W anchors (+k+1 label rows);
               labels are the request's own later tokens (lazy join).
               FIFO eviction by a monotonic counter (memory budget env).
  lora.py      zero-init LoRA on the MTP layer's DENSE linears via a
               per-layer quant_method proxy (any quant method), sharded to
               match TP (column: B sharded, row: A sharded) -> no extra
               collective at serve time; Megatron-style autograd collectives
               for training; replicated-factor grads all-reduced.
               Adapter tensors are persistent and updated only IN PLACE, so
               the drafter's captured CUDA graphs see new values.
  train.py     functional replay of the MTP layer over windows: frozen
               linears via quant_method.apply with probe-based dx backward,
               frozen routed experts via per-expert dequantized tiles (dx +
               d topk_w, no dW, no shadow bank), dense causal window
               attention, vocab-parallel CE; depth-1 teacher-forced CE on
               the committed tokens; AdamW over fp32 masters of the
               CANDIDATE adapter; micro-batch sized to SUFFIX_MTP_TUNE_STEP_MS.
  gate.py      replayed greedy draft chain (depths 1..k, the MTP's own
               recurrent hidden, serving's KV visibility) on held-out
               windows for live vs candidate (paired); promote when mean
               accepted length improves by >= margin over >= min anchors;
               per-language buckets (cheap umlaut / function-word tag).
  scheduler.py EngineCoreProc idle hook -> collective_rpc(worker_step) one
               micro-step at a time while has_work() is False (after a grace
               period), re-checking the input queue between micro-steps; all
               TP ranks run each micro-step in lockstep; rank 0 broadcasts
               the plan. Promotion = in-place copy on the serving stream
               inside the RPC (between engine steps). Any exception disables
               tuning for the process; a per-step liveness vote keeps the
               disable rank-uniform. Periodic "[suffix mtp-tune]" telemetry.

Env (all optional)
  SUFFIX_MTP_TUNE_RATE=0.25       fraction of requests captured
  SUFFIX_MTP_TUNE_WINDOW=256      anchors per window
  SUFFIX_MTP_TUNE_CTX=64          prompt rows kept as attention context
  SUFFIX_MTP_TUNE_MEM_GIB=4       pinned host arena (rank 0 only)
  SUFFIX_MTP_TUNE_MAX_ROWS=2048   captured rows per step (device staging)
  SUFFIX_MTP_TUNE_HELDOUT_PCT=20  requests reserved for the gate
  SUFFIX_MTP_TUNE_RANK=16 / _ALPHA=32 / _TARGETS=glob,glob,...
  SUFFIX_MTP_TUNE_LR=1e-4 / _ACCUM=4 / _STEP_MS=50 / _MAX_WINDOWS=4
  SUFFIX_MTP_TUNE_EVAL_EVERY=50   optimizer steps between gate evals
  SUFFIX_MTP_TUNE_EVAL_WINDOWS=64 / _MIN_EVAL_WINDOWS=8 / _MIN_TRAIN_WINDOWS=8
  SUFFIX_MTP_TUNE_MARGIN=0.05     accepted-length gain needed (tokens/step)
  SUFFIX_MTP_TUNE_MIN_ANCHORS=2000
  SUFFIX_MTP_TUNE_MIN_FREE_GIB=2  skip micro-steps below this free VRAM
  SUFFIX_MTP_TUNE_IDLE_GRACE_MS=200
  SUFFIX_MTP_TUNE_SHADOW=1        train + evaluate, never promote
  SUFFIX_MTP_TUNE_ROLLBACK_FILE=/tmp/suffix_mtp_tune.rollback  touch it ->
                                  live adapters zeroed (exact no-op) at the
                                  next idle micro-step, promotions off
  SUFFIX_MTP_TUNE_LOG_S=60 / _LANG=1 / _RPC_TIMEOUT_S=30
  SUFFIX_MTP_TUNE_FAULT_RANK=<r>  test only: raise inside the micro-step on rank r

Proven on CPU (tests/test_mtp_tune.py, shrunk Qwen4Exp-shaped MTP with
FP8-128-block fake-quantized experts): zero-init adapter = bitwise no-op;
frozen-expert and frozen-linear backward == autograd on dequantized
weights; TP=2 (gloo, 2 processes) sharded adapters + collectives give the
TP=1 forward and gradients; training on language-shifted windows raises
replayed acceptance and the gate promotes, already-fit data does not
promote; capture label join / gaps / eviction; engine idle hook yields;
in-place promotion is visible to a previously captured closure.
NOT proven (needs silicon): fidelity of the replay to the real vLLM
kernels (QSA sparse attention, fused HC kernels, FusedMoE numerics), the
CUDA-graph in-place visibility on real graphs, latency, memory, TP
collectives over NCCL, GLM.

Open risks
  R1 replay fidelity: window attention cannot see context before the
     window, QSA/DSA top-k sparsity is not replayed, FP8 act quant is
     emulated. Measured by parity_d1 (replay argmax vs the draft the server
     produced, live adapter); promotions are judged on replay, so a low
     parity means the gate measures the wrong thing.
  R2 a TP rank that raises INSIDE a collective sequence leaves the other
     ranks blocked in NCCL -> the engine hangs. Mitigated (same code path
     and shapes on every rank, uniform memory vote, liveness vote) but not
     eliminated. Must be fault-injected on the pod before prod.
  R3 the quant_method proxy: code doing isinstance/type checks on
     layer.quant_method after load would see the proxy.
  R4 serving cost of always-on adapters: 2 skinny GEMMs per adapted
     linear per draft step even when B == 0 (graphs cannot branch).
  R5 a request arriving mid-micro-step waits <= one micro-step
     (STEP_MS target + stream sync); p99 must be measured.
  R6 learning on sampled (temperature > 0) tokens and on only-depth-1
     loss; chain depths are only evaluated. KL to target top-k not done.
  R7 parity is diluted when SUFFIX_HYBRID_WRAP is also on (suffix rows
     replace native drafts); validate with WRAP off.
  R8 GLM: eh_proj runs through build_glm52_plan (bypasses Linear.forward
     -> must set _eh_plan = None when adapting it); kv_b_proj is absorbed
     by MLA decode (never adapt it); routed experts are NVFP4 (dequant
     not wired: experts_handle raises).

Pod validation protocol (qwen3.8-flash-next dev pod, TP=2+EP, k per prod)
  0. Build the bundle; set SUFFIX_MTP_TUNE=1 SUFFIX_MTP_TUNE_SHADOW=1
     SUFFIX_MTP_TUNE_LOG_S=30, SUFFIX_HYBRID_WRAP unset. Expect
     "[suffix mtp-tune] installed: N adapters ... tp=2" on both ranks and
     "engine idle hook installed". Serve smoke: outputs identical to a run
     without the gate (greedy, same prompts) - zero-init no-op on silicon.
  1. Baseline: same German + English benchmark as the 2026-09-29 GLM
     measurement (per-request tok/step from stream chunk sizes) with and
     without the gate (shadow): delta must be within noise (adapter GEMM
     cost R4) - record decode tok/s and acceptance per language.
  2. Capture: drive German traffic; check windows train/held grow,
     drops ~0, gaps small; host RSS growth ~= MEM_GIB.
  3. Parity: after the first eval line, parity_d1 must be >= 0.9 (else R1:
     fix the replay before trusting any promotion). Check parity per bucket.
  4. Idle training: loss decreases; max_micro_ms <= 2x STEP_MS;
     skip_mem stays 0; nvidia-smi peak VRAM during idle training vs the
     KV reservation.
  5. Latency: fire single requests at random times during idle training
     (e.g. 500 requests, Poisson); TTFT p50/p99 vs gate off; the log's
     max_added_ms must match; p99 delta must stay <= STEP_MS.
  6. Promotion: unset SHADOW; after a promotion re-run the benchmark:
     German tok/step up, English not down (gate is on the mixed held-out
     set, check per-language buckets too). Outputs still identical to the
     target-only greedy reference (drafts are verified).
  7. Rollback: touch the rollback file -> next idle micro-step logs
     ROLLBACK and acceptance returns to baseline.
  8. Faults: SUFFIX_MTP_TUNE_FAULT_RANK=1 -> both ranks log DISABLED, the
     engine keeps serving (R2). Also kill-test a promotion mid-traffic.
  GLM-5.3 (after the family exists): repeat 0-8 at TP=8 with NCCL split;
  additionally time the per-micro-step collectives (embedding all-reduce,
  row-parallel reduces, vocab-parallel CE) on PCIe (no P2P).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

GATE = "SUFFIX_MTP_TUNE"
DEFAULT_TARGETS = (
    # Qwen4Exp MTP (qwen3.8-flash-next): input fusion, attention, shared expert
    "*fc_embedding", "*fc_hidden", "*self_attn.qkv_proj", "*self_attn.o_proj",
    "*shared_expert.gate_up_proj", "*shared_expert.down_proj",
)


def _f(name, default):
    return float(os.environ.get(f"{GATE}_{name}", default))


def _i(name, default):
    return int(os.environ.get(f"{GATE}_{name}", default))


@dataclass
class Config:
    rate: float = 0.25
    window: int = 256
    ctx: int = 64
    mem_gib: float = 4.0
    max_rows: int = 2048
    heldout_pct: int = 20
    rank: int = 16
    alpha: float = 32.0
    targets: tuple = field(default_factory=lambda: DEFAULT_TARGETS)
    lr: float = 1e-4
    accum: int = 4
    step_ms: float = 50.0
    max_windows: int = 4
    eval_every: int = 50
    eval_windows: int = 64
    min_eval_windows: int = 8
    min_train_windows: int = 8
    margin: float = 0.05
    min_anchors: int = 2000
    min_free_gib: float = 2.0
    idle_grace_ms: float = 200.0
    shadow: bool = False
    rollback_file: str = "/tmp/suffix_mtp_tune.rollback"
    log_s: float = 60.0
    lang: bool = True
    rpc_timeout_s: float = 30.0

    @classmethod
    def from_env(cls):
        t = os.environ.get(f"{GATE}_TARGETS", "").strip()
        return cls(rate=_f("RATE", 0.25), window=_i("WINDOW", 256), ctx=_i("CTX", 64),
                   mem_gib=_f("MEM_GIB", 4), max_rows=_i("MAX_ROWS", 2048),
                   heldout_pct=_i("HELDOUT_PCT", 20), rank=_i("RANK", 16),
                   alpha=_f("ALPHA", 32),
                   targets=tuple(x.strip() for x in t.split(",") if x.strip()) or DEFAULT_TARGETS,
                   lr=_f("LR", 1e-4), accum=_i("ACCUM", 4), step_ms=_f("STEP_MS", 50),
                   max_windows=_i("MAX_WINDOWS", 4), eval_every=_i("EVAL_EVERY", 50),
                   eval_windows=_i("EVAL_WINDOWS", 64),
                   min_eval_windows=_i("MIN_EVAL_WINDOWS", 8),
                   min_train_windows=_i("MIN_TRAIN_WINDOWS", 8), margin=_f("MARGIN", 0.05),
                   min_anchors=_i("MIN_ANCHORS", 2000), min_free_gib=_f("MIN_FREE_GIB", 2),
                   idle_grace_ms=_f("IDLE_GRACE_MS", 200),
                   shadow=os.environ.get(f"{GATE}_SHADOW", "") == "1",
                   rollback_file=os.environ.get(f"{GATE}_ROLLBACK_FILE",
                                                "/tmp/suffix_mtp_tune.rollback"),
                   log_s=_f("LOG_S", 60), lang=os.environ.get(f"{GATE}_LANG", "1") == "1",
                   rpc_timeout_s=_f("RPC_TIMEOUT_S", 30))


def install():
    """sitecustomize entry (SUFFIX_MTP_TUNE=1): engine idle hook + V2
    runner load_model hook. Imports vLLM; never raises (logged refusal)."""
    import sys
    from suffix_hybrid.mtp_tune import scheduler
    cfg = Config.from_env()
    for fn in (scheduler.install_engine_hook, scheduler.install_worker_hook):
        try:
            fn(cfg)
        except Exception as exc:  # noqa: BLE001 - optional feature
            print(f"[suffix mtp-tune] {fn.__name__} failed (tuning off): "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
