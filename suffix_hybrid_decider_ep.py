"""Optional native vLLM EndpointPlugin; this seam is absent in Rust frontend.

This is an adapter inside vLLM's existing API server, not a separate inference
server. SystemOne validation, prompt construction, masking, calibration and
answer math are owned by Rust. Do not deploy with the stock Rust frontend:
it does not discover endpoint_plugins or support the pooling engine protocol.
"""
import json
import os
import uuid

from suffix_hybrid.decider_contract import (
    ENDPOINT_NAME, checkpoint_contract, validate_tokenizer)


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
            from vllm.pooling_params import PoolingParams
            context = request.app.state.suffix_decider
            try:
                body = await request.json()
                if isinstance(body, dict) and body.get("model") not in context["model_names"]:
                    return JSONResponse({"error": "Unknown decision model"}, status_code=404)
                body_json = json.dumps(body, ensure_ascii=False)
                prepared = json.loads(_native.decider_prepare(
                    body_json, json.dumps(context["saved"]["codes"])))
                logits, input_tokens = [], 0
                for row in prepared["rows"]:
                    token_ids = context["tokenizer"].apply_chat_template(
                        row["messages"], tokenize=True,
                        add_generation_prompt=True, enable_thinking=False)
                    if not token_ids or len(token_ids) > context["max_model_len"]:
                        raise ValueError("decision input must fit model limit without truncation")
                    input_tokens += len(token_ids)
                    final = None
                    context["engine"].check_admission()
                    async for result in context["engine"].encode(
                            {"prompt_token_ids": token_ids},
                            PoolingParams(task="classify", use_activation=False),
                            "decider-" + uuid.uuid4().hex):
                        final = result
                    if final is None:
                        raise RuntimeError("decision engine returned no pooling result")
                    logits.append(final.outputs.data.tolist())
                response = json.loads(_native.decider_answer(
                    body_json, json.dumps(logits), context["saved"]["temperature"]))
                response["usage"]["input_tokens"] = input_tokens
                print("SUFFIX_PPLX_DECIDER REQUEST completed rows=" + str(len(logits)), flush=True)
                return JSONResponse(response)
            except (ValueError, TypeError, KeyError) as error:
                return JSONResponse({"error": str(error)}, status_code=400)

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
        state.suffix_decider = {"engine": engine_client, "tokenizer": tokenizer,
                                "saved": saved,
                                "max_model_len": model_config.max_model_len,
                                "model_names": model_names(args, model_config)}


def model_names(args, model_config):
    names = getattr(args, "served_model_name", None) or [model_config.model]
    return [names] if isinstance(names, str) else list(names)


def register():
    return DecisionEndpointPlugin()
