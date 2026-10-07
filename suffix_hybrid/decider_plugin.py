"""Register a gated native vLLM Qwen3.5 pooling adapter (v0.30.0 only)."""
import os
from importlib.metadata import version


def register():
    if os.getenv("SUFFIX_PPLX_DECIDER") != "1":
        return
    if version("vllm") != "0.30.0":
        raise RuntimeError("pplx-decider adapter is pinned to vLLM 0.30.0")
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP, Qwen3_5ForConditionalGenerationConfig)
    from suffix_hybrid.decider_contract import (
        ARCHITECTURE, configure_model, validate_engine)

    class DecisionConfig(Qwen3_5ForConditionalGenerationConfig):
        verify_and_update_model_config = staticmethod(configure_model)

        @staticmethod
        def verify_and_update_config(vllm_config):
            Qwen3_5ForConditionalGenerationConfig.verify_and_update_config(vllm_config)
            validate_engine(vllm_config)

    MODELS_CONFIG_MAP[ARCHITECTURE] = DecisionConfig
    ModelRegistry.register_model(
        ARCHITECTURE, "suffix_hybrid.decider_model:PplxDeciderForSequenceClassification")
    print("SUFFIX_PPLX_DECIDER REGISTERED native Qwen3.5/LAST/255-head; vLLM=0.30.0", flush=True)
