"""Register a gated native vLLM Qwen3.5 pooling adapter (validated vLLM releases only)."""
import os
from importlib.metadata import version


# Releases whose Qwen3.5/Qwen3-VL, pooling, model-state and endpoint anchors
# this adapter was checked against. Anything else refuses to start.
VALIDATED_VLLM = ("0.30.0", "0.32.0")


def register():
    if os.getenv("SUFFIX_PPLX_DECIDER") != "1":
        return
    vllm_version = version("vllm")
    if vllm_version not in VALIDATED_VLLM:
        raise RuntimeError(f"pplx-decider adapter is pinned to vLLM {VALIDATED_VLLM}, got {vllm_version}")
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
            # vLLM 0.30/0.32 resolve quant_config after this hook. Check serialized
            # metadata/invariants here; the model constructor checks the
            # resolved native format before any backbone allocation.
            validate_engine(vllm_config, require_resolved_quantization=False)

    MODELS_CONFIG_MAP[ARCHITECTURE] = DecisionConfig
    ModelRegistry.register_model(
        ARCHITECTURE, "suffix_hybrid.decider_model:PplxDeciderForSequenceClassification")
    print(f"SUFFIX_PPLX_DECIDER REGISTERED native Qwen3.5/LAST/255-head; vLLM={vllm_version}", flush=True)
