"""Compose vLLM's native encoder and GDN state paths without replacing kernels.

The stock V2 resolver picks EncoderOnlyModelState before MambaHybridModelState.
Cooperative MRO retains the latter's recurrent-state prep and adds the former's
noncausal full-attention metadata. Native methods all use super(); no duplicate
DefaultModelState construction and no change to causal GDN mathematics.
"""
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class DecisionHybridModelState(EncoderOnlyModelState, MambaHybridModelState):
    pass
