"""Inkling 2 loop checkpoints (thinkingmachines/Inkling-2-Mini) on a vLLM build that only ships Inkling 2 Small.

Port of vllm-project/vllm#69028 at fc1443152c: that head's whole `vllm/models/inkling2`
package lives in `suffix_hybrid/inkling2` (imports rewritten) and serves the loop
architecture and its DSpark drafter. The nightly's own package keeps serving
nothing else here: this is only armed on the Inkling pool (SUFFIX_INKLING2_LOOP=1).

On SM120 the PR's changed CUDA ops (conv_mix, all-reduce, reduce-scatter) are not
reached: conv_mix takes the Triton short_conv + attn_res path there and TP=1 has no
collective, so the port needs no compiled code. Every hook below checks the exact
nightly source it replaces and refuses on drift.
"""
import inspect
import os
from importlib.metadata import version

# Nightly commit f8b29fab (2026-11-21): Inkling 2 Small is in, the loop is not.
VALIDATED_VLLM = ("0.34.0rc1.dev1108+gf8b29fab4",)
LOOP_ARCH = "Inkling2LoopForConditionalGeneration"


def _anchor(obj, needle):
    src = inspect.getsource(obj)
    if needle not in src:
        raise RuntimeError(f"SUFFIX_INKLING2_LOOP: anchor drifted in {obj.__qualname__}: {needle!r} not found")


def register():
    if os.getenv("SUFFIX_INKLING2_LOOP") != "1":
        return
    vllm_version = version("vllm")
    if vllm_version not in VALIDATED_VLLM:
        raise RuntimeError(f"SUFFIX_INKLING2_LOOP is pinned to vLLM {VALIDATED_VLLM}, got {vllm_version}")

    from vllm.model_executor.models import ModelRegistry
    ModelRegistry.register_model(LOOP_ARCH, f"suffix_hybrid.inkling2:{LOOP_ARCH}")
    # The drafter must read the plugin's DSpark config (loop-aware target_layer_ids).
    ModelRegistry.register_model("Inkling2DSparkModel", "suffix_hybrid.inkling2:Inkling2DSparkModel")

    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from suffix_hybrid.inkling2.configs import Inkling2DSparkConfig, Inkling2LoopMMConfig
    if "inkling2" not in _CONFIG_REGISTRY:
        raise RuntimeError("SUFFIX_INKLING2_LOOP: this vLLM has no inkling2 config registry entry")
    _CONFIG_REGISTRY["inkling2_loop"] = Inkling2LoopMMConfig  # LazyConfigDict returns a type as-is

    # Quantization: the nightly's override only knows the Small architecture, so a loop
    # checkpoint would fall through to plain ModelOpt. Same class, wider match.
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig
    _anchor(Inkling2QuantConfig.override_quantization_method, '"Inkling2ForConditionalGeneration" in architectures')

    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        archs = set(getattr(hf_config, "architectures", None) or [])
        return "inkling2" if archs & {"Inkling2ForConditionalGeneration", LOOP_ARCH} else None
    Inkling2QuantConfig.override_quantization_method = classmethod(override_quantization_method)

    # Speculative decoding: DSpark's config is built from the target's config.
    from vllm.config.speculative import SpeculativeConfig
    _anchor(SpeculativeConfig.hf_config_override, 'if hf_config.model_type == "inkling2":')
    original_override = SpeculativeConfig.hf_config_override

    def hf_config_override(hf_config, *args, **kwargs):
        if getattr(hf_config, "model_type", None) == "inkling2_loop":
            return Inkling2DSparkConfig.from_target(hf_config)
        return original_override(hf_config, *args, **kwargs)
    SpeculativeConfig.hf_config_override = staticmethod(hf_config_override)

    # Chat rendering: the loop shares Inkling 2's reference renderer.
    from vllm.renderers import inkling as renderer
    _anchor(renderer.InklingRenderer.__init__, '"model_type", None) == "inkling2"')
    original_init = renderer.InklingRenderer.__init__

    def __init__(self, config, tokenizer):
        original_init(self, config, tokenizer)
        if getattr(self.model_config.hf_config, "model_type", None) == "inkling2_loop":
            self._is_inkling2 = True
            self._inkling_tokenizer = renderer._HfBackedTmlTokenizer(self.get_tokenizer(), split_special_tokens=True)
    renderer.InklingRenderer.__init__ = __init__

    # Breakable CUDA graphs, as for the Small architecture.
    import vllm.config.vllm as vllm_config
    if "Inkling2ForConditionalGeneration" not in vllm_config.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES:
        raise RuntimeError("SUFFIX_INKLING2_LOOP: Inkling 2 is not a breakable-cudagraph architecture in this vLLM")
    vllm_config.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES = vllm_config.DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES | {LOOP_ARCH}

    print(f"SUFFIX_INKLING2_LOOP REGISTERED {LOOP_ARCH} (vllm#69028 @ fc1443152c); vLLM={vllm_version}", flush=True)
