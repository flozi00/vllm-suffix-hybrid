# SPDX-License-Identifier: Apache-2.0
"""The nightly's Inkling2QuantConfig, shared: vLLM builds the quant config from its
own `inkling2` method, so the model's isinstance check must see that class.
suffix_hybrid.inkling2_plugin widens its architecture match to the loop."""
from vllm.models.inkling2.quant_config import Inkling2QuantConfig

__all__ = ["Inkling2QuantConfig"]
