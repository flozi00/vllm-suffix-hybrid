# SPDX-License-Identifier: Apache-2.0
"""In-place LoRA SFT of the RUNNING vLLM model, served by the same weights.

Gate: SUFFIX_SFT=1 (default OFF; unset = nothing imported or patched).
HTTP routes need the endpoint plugin allowlisted: VLLM_PLUGINS must name
``suffix_sft`` (and, because VLLM_PLUGINS also filters general plugins,
every suffix_* kernel plugin the pod uses).

What "in place" means: there is ONE model. Zero-init LoRA adapters are
attached to the served language model's linears at load time (before
torch.compile / CUDA-graph capture), so every request already runs base +
adapter. Training runs the same live modules (same NVFP4 kernels, same TP
shards) with autograd swapped in (sft/model.py) and, when a run succeeds,
copies the trained adapter values into the live adapter tensors in place:
the next serving step uses them. No second model, no weight copy, no
reload; prefix cache is reset so no KV from the old weights is reused.

Pieces
  model.py   serving seam (quant-method subclass adds the delta; adapters
             are module buffers) + TrainMode (autograd over the live
             modules: frozen-linear Function, TP f/g collectives, SDPA per
             packed segment, GDN conv + gated delta rule, native norms /
             RoPE, layer checkpointing with host-offloaded inputs, chunked
             vocab-parallel CE) + the gradient-reach guard
  worker.py  Worker.load_model hook + ``suffix_sft`` RPC: begin / micro /
             step / eval / save / commit / abort / load / reset; flat bf16
             working params on the GPU, fp32 master + AdamW on the host
  api.py     /v1/sft/train, /v1/sft/load, /v1/sft/status; chat-template
             tokenization (loss on the final assistant turn), packing,
             served-vs-trained parity, private Hugging Face Hub checkpoints
             with a latest.json pointer resumed at startup (parakeet-server's
             OnlineTrainer contract)

Scheduling: one RPC per micro-batch / optimizer step. execute_model and RPCs
share the worker queue, so a training op never overlaps a serving step;
between ops the API yields to serving whenever requests are in flight
(SUFFIX_SFT_DUTY = training's share of GPU time under load).

Env (all optional)
  SUFFIX_SFT_RANK=16 / _ALPHA=32 / _TARGETS=glob,glob  (LM-relative names)
  SUFFIX_SFT_MAX_TOKENS=4096    tokens per micro-batch (= longest example)
  SUFFIX_SFT_MIN_FREE_GIB=4     refuse a micro-step below this free VRAM
  SUFFIX_SFT_OFFLOAD_GIB=2      pinned host budget for checkpointed layer
                                inputs per micro-step (0 = keep on GPU); the
                                pod's RAM limit also holds vLLM + the master /
                                AdamW state (12 B per adapter param)
  SUFFIX_SFT_DUTY=0.5           training share of GPU time while serving
  SUFFIX_SFT_MAX_EXAMPLES=256 / _MAX_STEPS=1000
  SUFFIX_SFT_REPO_ID=<org/name> private HF model repo for checkpoints
                                (HF_TOKEN needed); unset = local only
  SUFFIX_SFT_DIR=$HF_HOME/sft   local checkpoint root
  SUFFIX_SFT_PARITY_NATS=0.15   max |served - trained| mean NLL before a run
  SUFFIX_SFT_RPC_TIMEOUT_S=600

Supported: dense attention + GDN hybrids (Qwen3.5 / qwen-fable, Qwen3-Next
dense), Gemma-4 dense (sliding + full, KV sharing), LFM2-MoE (short-conv, FP8
experts frozen, sigmoid + bias router), Qwen4Exp, TP >= 1. Refused at
install with a logged reason (serving unaffected): MLA, multi-group routing,
PP, soft-capped attention, other recurrent layers. Vision towers are never
touched (text-only SFT; image inputs keep working).

Risks / not proven on silicon (CPU tests: tests/test_sft.py)
  R1 serving cost: two skinny GEMMs per adapted linear per step (B == 0
     still costs; graphs cannot branch), and the bf16 input of every
     adapted linear now has a second reader, so Inductor norm/act+quant
     fusions into the GEMM may no longer match (numerics differ slightly
     from SFT off even at B == 0). Measure decode tok/s vs SFT off.
  R2 a rank raising INSIDE a collective leaves the others blocked (memory
     vote + liveness vote first; not eliminated). Fault-inject before prod.
  R3 the training forward's fidelity to serving (fused paths we disabled,
     KV-cache quantization, FlashAttention vs SDPA) is MEASURED per run
     (parity_nats) and the run refused above SUFFIX_SFT_PARITY_NATS.
  R4 MTP drafts do not see the adapter: acceptance drops as it diverges.
  R5 serving pauses for the length of each RPC (bounded by MAX_TOKENS).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

GATE = "SUFFIX_SFT"


def _env(name, default):
    return os.environ.get(f"{GATE}_{name}", "").strip() or default


@dataclass
class Config:
    rank: int = 16
    alpha: float = 32.0
    rslora: bool = False
    lora_plus: float = 1.0
    targets: tuple = field(default_factory=tuple)
    max_tokens: int = 4096
    min_free_gib: float = 4.0
    offload_gib: float = 2.0
    mlp_tile: int = 4096
    offload_saved: bool = False
    mem_trace: bool = False
    host_ckpt: bool = False
    swap_keep: int = 0
    saved_min_mb: int = 1
    free_kv: bool = False
    duty: float = 0.5
    max_examples: int = 256
    max_steps: int = 1000
    repo_id: str = ""
    dir: str = ""
    parity_nats: float = 0.15
    rpc_timeout_s: float = 600.0

    @classmethod
    def from_env(cls):
        from suffix_hybrid.sft.model import DEFAULT_TARGETS
        t = tuple(x.strip() for x in _env("TARGETS", "").split(",") if x.strip())
        home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        return cls(rank=int(_env("RANK", 16)), alpha=float(_env("ALPHA", 32)),
                   rslora=bool(int(_env("RSLORA", 0))),
                   lora_plus=float(_env("LORA_PLUS", 1.0)),
                   targets=t or DEFAULT_TARGETS, max_tokens=int(_env("MAX_TOKENS", 4096)),
                   min_free_gib=float(_env("MIN_FREE_GIB", 4)),
                   offload_gib=float(_env("OFFLOAD_GIB", 2)),
                   mlp_tile=int(_env("MLP_TILE", 4096)),
                   offload_saved=bool(int(_env("OFFLOAD_SAVED", 0))),
                   mem_trace=bool(int(_env("MEM_TRACE", 0))),
                   host_ckpt=bool(int(_env("HOST_CKPT", 0))),
                   swap_keep=int(_env("SWAP_KEEP", 0)),
                   saved_min_mb=int(_env("SAVED_MIN_MB", 1)),
                   free_kv=bool(int(_env("FREE_KV", 0))),
                   duty=min(1.0, max(0.05, float(_env("DUTY", 0.5)))),
                   max_examples=int(_env("MAX_EXAMPLES", 256)),
                   max_steps=int(_env("MAX_STEPS", 1000)), repo_id=_env("REPO_ID", ""),
                   dir=_env("DIR", os.path.join(home, "sft")),
                   parity_nats=float(_env("PARITY_NATS", 0.15)),
                   rpc_timeout_s=float(_env("RPC_TIMEOUT_S", 600)))


TARGET = "vllm.v1.worker.gpu_worker"


def install():
    """sitecustomize entry (SUFFIX_SFT=1): patch the GPU Worker when its
    module loads (worker processes only import it). Never raises."""
    # Stale-artifact guard (as SUFFIX_ROCM_HC_FUSE): an AOT artifact compiled
    # without the adapters would load without retracing and serve base only.
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"
    import importlib.util

    def patch(mod):
        try:
            from suffix_hybrid.sft.worker import patch_worker
            patch_worker(mod, Config.from_env())
        except Exception as exc:  # noqa: BLE001 - optional feature
            print(f"[suffix sft] worker patch failed (SFT off): {exc!r}", file=sys.stderr, flush=True)

    if TARGET in sys.modules:
        patch(sys.modules[TARGET])
        return

    class _Finder:
        _suffix_sft = True

        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname != TARGET:
                return None
            sys.meta_path[:] = [f for f in sys.meta_path if f is not self]
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            real = spec.loader.exec_module

            def exec_module(module):
                real(module)
                patch(module)

            spec.loader.exec_module = exec_module  # type: ignore[method-assign]
            return spec

    if not any(getattr(f, "_suffix_sft", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
