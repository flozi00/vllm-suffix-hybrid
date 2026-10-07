"""Small fail-closed checkpoint/config adapter; no inference computation."""
import json
import math
from pathlib import Path

ARCHITECTURE = "PplxDeciderForSequenceClassification"
PLUGIN_NAME = "suffix_pplx_decider"
ENDPOINT_NAME = "suffix_pplx_decider_endpoint"


def checkpoint_contract(model_path):
    root = Path(model_path)
    if not root.is_dir():
        raise ValueError("pplx-decider requires a staged local checkpoint directory")
    saved = json.loads((root / "decision_config.json").read_text())
    if (saved.get("format_version") != 1
            or saved.get("attention_mode") != "noncausal_full_attention"
            or saved.get("pooling") != "last"):
        raise ValueError("expected v1.1 noncausal_full_attention/LAST decision checkpoint")
    codes, token_ids = saved.get("codes"), saved.get("token_ids")
    if (not isinstance(codes, list) or len(codes) != 255
            or any(not isinstance(code, str) or not code for code in codes)
            or len(set(codes)) != 255
            or not isinstance(token_ids, list) or len(token_ids) != 255
            or any(type(token) is not int or token < 0 for token in token_ids)
            or len(set(token_ids)) != 255):
        raise ValueError("decision checkpoint must declare 255 unique codes/token IDs")
    temperature = saved.get("temperature")
    if (type(temperature) not in (int, float)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("decision checkpoint temperature must be finite and positive")
    if not (root / "readout.safetensors").is_file():
        raise ValueError("decision checkpoint is missing readout.safetensors")
    return saved


def validate_tokenizer(tokenizer, saved):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Choose an option."}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    for code, token in zip(saved["codes"], saved["token_ids"]):
        if (tokenizer.encode(code, add_special_tokens=False) != [token]
                or tokenizer.encode(prefix + code, add_special_tokens=False)
                != prefix_ids + [token]):
            raise ValueError("checkpoint decision codes differ from tokenizer")


def validate_engine(vllm_config):
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    parallel = vllm_config.parallel_config
    pooler = model.pooler_config
    if model.runner_type != "pooling" or pooler is None:
        raise ValueError("pplx-decider requires --runner pooling")
    if pooler.seq_pooling_type != "LAST":
        raise ValueError("pplx-decider requires LAST pooling")
    if model.max_model_len > 8192:
        raise ValueError("pplx-decider's validated input limit is 8192 tokens")
    if cache.enable_prefix_caching or scheduler.enable_chunked_prefill:
        raise ValueError("noncausal decisions forbid prefix caching/chunked prefill")
    if cache.mamba_cache_mode != "none":
        raise ValueError("decision requests require fresh GDN state (--mamba-cache-mode none)")
    if cache.mamba_ssm_cache_dtype != "float32":
        raise ValueError("decision checkpoint requires float32 GDN recurrent state")
    if scheduler.max_num_batched_tokens < model.max_model_len:
        raise ValueError("noncausal decisions require one complete prefill per request")
    if vllm_config.speculative_config is not None:
        raise ValueError("decision pooling does not use speculative decoding")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("pplx-decider plugin requires model runner V2")
    if (parallel.pipeline_parallel_size != 1 or parallel.tensor_parallel_size != 1
            or parallel.enable_dbo or getattr(parallel, "ubatch_size", 1) != 1):
        raise ValueError("decision adapter currently supports TP=1, PP=1 and no DBO")
    if not model.enforce_eager:
        raise ValueError("decision adapter requires --enforce-eager until capture parity passes")
    if model.dtype.__str__() != "torch.bfloat16":
        raise ValueError("decision backbone and readout require bfloat16 compute")
    if getattr(model, "quantization", None) not in (None, "fp8"):
        raise ValueError("decision adapter currently supports BF16 or upstream FP8")


def configure_model(model_config):
    saved = checkpoint_contract(model_config.model)
    for config in (model_config.hf_config, model_config.hf_text_config):
        config.is_causal = False
        config.num_labels = 255
        config.problem_type = "single_label_classification"
    if model_config.hf_text_config.hidden_size != 5120:
        raise ValueError("unsupported decision readout hidden size")
    model_config._suffix_decider_contract = saved
