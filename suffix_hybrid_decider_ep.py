"""Optional native vLLM EndpointPlugin; this seam is absent in Rust frontend.

This is an adapter inside vLLM's existing API server, not a separate inference
server. SystemOne validation, prompt construction, masking, calibration and
answer math are owned by Rust. Do not deploy with the stock Rust frontend:
it does not discover endpoint_plugins or support the pooling engine protocol.
"""
import asyncio
import json
import logging
import os
import uuid

from suffix_hybrid.decider_contract import (
    ENDPOINT_NAME, checkpoint_contract, media_enabled, validate_tokenizer)

logger = logging.getLogger(__name__)


class DecisionEndpointPlugin:
    name = ENDPOINT_NAME
    required_tasks = ("classify",)

    def attach_router(self, app):
        if os.getenv("SUFFIX_PPLX_DECIDER") != "1":
            return
        from fastapi import Request
        from fastapi.responses import JSONResponse

        async def systemone(request: Request):
            from suffix_hybrid import _native
            from vllm.inputs.engine import tokens_input
            from vllm.pooling_params import PoolingParams
            context = request.app.state.suffix_decider
            try:
                body = await request.json()
                if isinstance(body, dict) and body.get("model") not in context["model_names"]:
                    return JSONResponse({"error": "Unknown decision model"}, status_code=404)
                body_json = json.dumps(body, ensure_ascii=False)
                prepared = json.loads(_native.decider_prepare(
                    body_json, json.dumps(context["saved"]["codes"]),
                    context["saved"]["profile"]))
                images, videos = body.get("images") or [], body.get("videos") or []
                mm = await asyncio.to_thread(decode_media, context, images, videos)
                prompts, input_tokens = [], 0
                for row in prepared["rows"]:
                    rendered = context["tokenizer"].apply_chat_template(
                        row["messages"], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
                    if not isinstance(rendered, str):
                        raise ValueError("decision chat template must render text")
                    # Transformers versions differ in the default tokenized
                    # chat-template return shape. Encode the exact rendered
                    # prefix, as the checkpoint's publisher does, instead of
                    # accidentally passing BatchEncoding dictionary keys.
                    token_ids = context["tokenizer"].encode(
                        rendered, add_special_tokens=False)
                    if not isinstance(token_ids, list) or any(
                            type(token) is not int or token < 0 for token in token_ids):
                        raise ValueError("decision tokenizer must return integer token IDs")
                    if not token_ids or len(token_ids) > context["max_model_len"]:
                        raise ValueError("decision input must fit model limit without truncation")
                    if context.get("pad_ids") and tuple(map(token_ids.count, context["pad_ids"])) != (
                            len(images), len(videos)):
                        raise ValueError("the state or question contains a literal image or "
                                         "video placeholder token")
                    # Media rows go through vLLM's raw-prompt path: its Qwen3-VL
                    # processor expands each placeholder with the d3 budgets.
                    prompts.append((token_ids, {"prompt_token_ids": token_ids, "multi_modal_data": mm}
                                    if mm else tokens_input(token_ids)))
            except (ValueError, TypeError, KeyError) as error:
                return JSONResponse({"error": str(error)}, status_code=400)

            # Inputs are already rendered with the checkpoint's exact chat
            # template. Native EngineInput bypasses the deprecated raw-prompt
            # renderer path and preserves those token IDs without re-rendering.
            try:
                logits = []
                for token_ids, prompt in prompts:
                    final = None
                    context["engine"].check_admission()
                    async for result in context["engine"].encode(
                            prompt,
                            PoolingParams(task="classify", use_activation=False),
                            "decider-" + uuid.uuid4().hex):
                        final = result
                    if final is None:
                        raise RuntimeError("decision engine returned no pooling result")
                    logits.append(final.outputs.data.tolist())
                    # Expanded length (image/video tokens included) when vLLM reports it.
                    input_tokens += len(getattr(final, "prompt_token_ids", None) or token_ids)
                response = json.loads(_native.decider_answer(
                    body_json, json.dumps(logits), context["saved"]["temperature"],
                    context["saved"]["profile"]))
                response["usage"]["input_tokens"] = input_tokens
                print("SUFFIX_PPLX_DECIDER REQUEST completed rows=" + str(len(logits)), flush=True)
                return JSONResponse(response)
            except Exception as error:
                # vLLM's own request errors (e.g. media tokens over the model
                # length) are the client's, as in vLLM's API server.
                if (rejected := client_error(error)) is not None:
                    return JSONResponse({"error": str(rejected)}, status_code=400)
                # EngineGenerateError may have an empty str() while retaining
                # its actual failure in __cause__. exc_info logs that entire
                # chain. Unexpected engine failures remain server failures.
                logger.exception("SUFFIX_PPLX_DECIDER unexpected inference failure")
                raise

        app.add_api_route("/v1/systemone", systemone, methods=["POST"],
                          name="suffix_pplx_decider_systemone")
        print("SUFFIX_PPLX_DECIDER ENDPOINT /v1/systemone attached", flush=True)

    async def init_state(self, engine_client, state, args):
        if os.getenv("SUFFIX_PPLX_DECIDER") != "1":
            return
        if engine_client is None:
            raise RuntimeError("decision endpoint requires an existing vLLM engine")
        from suffix_hybrid import _native
        if not all(hasattr(_native, name) for name in ("decider_prepare", "decider_answer")):
            raise RuntimeError("decision runtime bundle lacks the Rust helpers")
        model_config = engine_client.model_config
        saved = checkpoint_contract(model_config.model)
        tokenizer = engine_client.renderer.get_tokenizer()
        validate_tokenizer(tokenizer, saved)
        media = None
        if media_enabled(model_config, saved):
            from suffix_hybrid import decider_media
            media = {"video_processor": decider_media.video_processor(model_config.model)}
        pad_ids = (tuple(tokenizer.convert_tokens_to_ids(t) for t in ("<|image_pad|>", "<|video_pad|>"))
                   if saved["profile"] == "d3" else None)
        state.suffix_decider = {"engine": engine_client, "tokenizer": tokenizer,
                                "saved": saved, "media": media, "pad_ids": pad_ids,
                                "max_model_len": model_config.max_model_len,
                                "model_names": model_names(args, model_config)}


def decode_media(context, images, videos):
    """d3 images/videos -> vLLM multi_modal_data (blocking decode, run off the loop)."""
    if not images and not videos:
        return None
    if context.get("media") is None:
        raise ValueError("this deployment reads text only; images and videos are unsupported")
    from suffix_hybrid import decider_media
    mm = {}
    for name, items, load in (("image", images, decider_media.load_image),
                              ("video", videos, decider_media.load_video)):
        decoded = []
        for number, value in enumerate(items):
            try:
                decoded.append(load(value))
            except ValueError as error:
                raise ValueError(f"{name}s[{number}]: {error}") from error
        if decoded:
            mm[name] = decoded
    if videos:
        decider_media.check_video_budget(context["media"]["video_processor"], mm["video"])
    return mm


def client_error(error):
    """The vLLM client error in an exception chain, if any."""
    try:
        from vllm.exceptions import VLLMClientError
    except ImportError:
        return None
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, VLLMClientError):
            return error
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return None


def model_names(args, model_config):
    names = getattr(args, "served_model_name", None) or [model_config.model]
    return [names] if isinstance(names, str) else list(names)


def register():
    return DecisionEndpointPlugin()
