"""Reuse vLLM's Qwen3.5 backbone and native linear/pooling implementations."""
import torch
from safetensors import safe_open
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.pooler.seqwise import (
    ClassifierPoolerHead, LastPool, SequencePooler)
from vllm.model_executor.layers.pooler.activations import PoolerClassify
from vllm.model_executor.models.adapters import _create_pooling_model_cls
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from suffix_hybrid.decider_contract import checkpoint_contract, validate_engine


class _FloatReadout(ReplicatedLinear):
    def forward(self, hidden_states):
        return super().forward(hidden_states).float()


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder)
class PplxDeciderForSequenceClassification(
        _create_pooling_model_cls(Qwen3_5ForConditionalGeneration)):
    # HF Qwen3_5Model exports bare language_model.* and visual.*. The upstream
    # wrapper stores the text backbone under language_model.model.*.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "language_model.": "language_model.model.", "mtp.": None})

    def __init__(self, *, vllm_config, prefix=""):
        validate_engine(vllm_config)
        self._decision_saved = checkpoint_contract(vllm_config.model_config.model)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        print("SUFFIX_PPLX_DECIDER MODEL native hybrid backbone; noncausal full attention; BF16 readout", flush=True)

    @staticmethod
    def get_model_state_cls():
        from suffix_hybrid.decider_model_state import DecisionHybridModelState
        return DecisionHybridModelState

    def _init_pooler(self, vllm_config, prefix=""):
        # The generic pooling wrapper suppresses creation of the full-vocabulary
        # LM head. FP8 quantization stays on upstream backbone linears, never
        # this small 255-row trained decision head.
        self.readout = _FloatReadout(
            5120, 255, bias=False, params_dtype=torch.bfloat16,
            quant_config=None, return_bias=False, prefix="readout")
        return SequencePooler(
            pooling=LastPool(),
            head=ClassifierPoolerHead(
                classifier=self.readout, head_dtype=torch.bfloat16,
                activation=PoolerClassify(num_labels=255)))

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # The readout is separate from the backbone safetensors index. Load it
        # explicitly, fail closed on wrong shape/dtype/keys, and tell vLLM's
        # missing-parameter checker that this parameter was initialized.
        path = self.vllm_config.model_config.model + "/readout.safetensors"
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            if list(checkpoint.keys()) != ["weight"]:
                raise ValueError("unsupported decision readout tensors")
            weight = checkpoint.get_tensor("weight")
        if tuple(weight.shape) != (255, 5120) or weight.dtype != torch.bfloat16:
            raise ValueError("expected BF16 255x5120 decision readout")
        default_weight_loader(self.readout.weight, weight)
        loaded.add("readout.weight")
        print("SUFFIX_PPLX_DECIDER READOUT loaded 255x5120 BF16", flush=True)
        return loaded
