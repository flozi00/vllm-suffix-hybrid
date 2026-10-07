"""Exercise native endpoint boundary without requiring a GPU or vLLM install."""
import json
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

try:
    import suffix_hybrid
except ModuleNotFoundError as error:
    if error.name != "suffix_hybrid":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import suffix_hybrid

# This root-level endpoint module ships in runtime_bundle, not the native
# wheel. Load that adapter explicitly without shadowing the installed wheel
# package used by every other pytest module in Linux CI.
endpoint_spec = importlib.util.spec_from_file_location(
    "suffix_hybrid_decider_ep_test", Path(__file__).resolve().parents[1]
    / "suffix_hybrid_decider_ep.py")
endpoint_module = importlib.util.module_from_spec(endpoint_spec)
endpoint_spec.loader.exec_module(endpoint_module)
DecisionEndpointPlugin = endpoint_module.DecisionEndpointPlugin
model_names = endpoint_module.model_names


class Response:
    def __init__(self, content, status_code=200):
        self.content, self.status_code = content, status_code


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.token_input_factory_calls = []
        owner = self
        class Engine:
            def check_admission(self):
                owner.calls.append("admitted")
            async def encode(self, prompt, params, request_id):
                owner.calls.append((prompt, params))
                yield NS(outputs=NS(data=NS(tolist=lambda: [3.0, 2.0] + [1.0] * 253)))
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                owner.calls.append(kwargs)
                return [9, 8, 7]
        self.context = {"engine": Engine(), "tokenizer": Tokenizer(),
                        "saved": {"codes": ["A", "B"], "temperature": 1.0087417621345625},
                        "max_model_len": 8192, "model_names": ["decider"]}
        self.app = NS(state=NS(suffix_decider=self.context))
        self.app.add_api_route = lambda path, handler, **kwargs: setattr(self, "handler", handler)
        fastapi = ModuleType("fastapi")
        fastapi.Request = NS
        responses = ModuleType("fastapi.responses")
        responses.JSONResponse = Response
        pooling = ModuleType("vllm.pooling_params")
        pooling.PoolingParams = lambda **kwargs: NS(**kwargs)
        inputs = ModuleType("vllm.inputs.engine")
        def tokens_input(token_ids):
            self.token_input_factory_calls.append(token_ids)
            return {"type": "token", "prompt_token_ids": token_ids}
        inputs.tokens_input = tokens_input
        self.modules = {"fastapi": fastapi, "fastapi.responses": responses,
                        "vllm.pooling_params": pooling, "vllm.inputs.engine": inputs}
        self.native = NS(decider_prepare=self.prepare, decider_answer=self.answer)

    def prepare(self, request_json, codes_json):
        return json.dumps({"rows": [{"messages": [{"role": "user", "content": "native prompt"}],
                                     "option_count": 2}]})

    def answer(self, request_json, logits_json, temperature):
        self.calls.append((json.loads(logits_json), temperature))
        return json.dumps({"model": "decider", "answers": {},
                           "usage": {"input_tokens": 0, "output_tokens": 0}})

    async def invoke(self, body):
        async def request_json():
            return body
        with patch.dict(os.environ, {"SUFFIX_PPLX_DECIDER": "1"}):
            with patch.dict(sys.modules, self.modules):
                with patch.object(suffix_hybrid, "_native", self.native, create=True):
                    DecisionEndpointPlugin().attach_router(self.app)
                    return await self.handler(NS(app=self.app, json=request_json))

    async def test_unknown_model_is_rejected_without_engine_work(self):
        result = await self.invoke({"model": "other"})
        self.assertEqual(result.status_code, 404)
        self.assertEqual(self.calls, [])

    async def test_raw_logits_reach_native_calibration_once_and_usage_is_real(self):
        result = await self.invoke({"model": "decider"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content["usage"]["input_tokens"], 3)
        self.assertEqual(result.content["usage"]["output_tokens"], 0)
        prompt, params = self.calls[2]
        self.assertEqual(prompt["prompt_token_ids"], [9, 8, 7])
        self.assertEqual(prompt["type"], "token")
        self.assertEqual(self.token_input_factory_calls, [[9, 8, 7]])
        self.assertIs(prompt["prompt_token_ids"], self.token_input_factory_calls[0])
        self.assertFalse(params.use_activation)
        self.assertEqual(params.task, "classify")
        raw_logits, temperature = self.calls[-1]
        self.assertEqual(raw_logits[0][:2], [3.0, 2.0])
        self.assertEqual(temperature, self.context["saved"]["temperature"])

    async def test_oversized_input_is_rejected_without_truncation(self):
        self.context["max_model_len"] = 2
        result = await self.invoke({"model": "decider"})
        self.assertEqual(result.status_code, 400)
        self.assertNotIn("admitted", self.calls)

    async def test_native_validation_failure_is_client_error(self):
        self.native.decider_prepare = lambda *args: (_ for _ in ()).throw(ValueError("images unsupported"))
        result = await self.invoke({"model": "decider", "images": ["x"]})
        self.assertEqual(result.status_code, 400)
        self.assertEqual(self.calls, [])

    async def test_unexpected_wrapped_engine_error_logs_cause_and_propagates(self):
        async def encode(*args):
            try:
                raise ValueError("engine input failure witness")
            except ValueError as cause:
                raise RuntimeError() from cause
            yield
        self.context["engine"].encode = encode
        with self.assertLogs(endpoint_module.logger, level="ERROR") as captured:
            with self.assertRaises(RuntimeError) as raised:
                await self.invoke({"model": "decider"})
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("engine input failure witness", "\n".join(captured.output))
        self.assertIn("Traceback", "\n".join(captured.output))

    async def test_engine_value_error_is_not_misreported_as_bad_client_input(self):
        async def encode(*args):
            raise ValueError("unexpected engine failure")
            yield
        self.context["engine"].encode = encode
        with self.assertLogs(endpoint_module.logger, level="ERROR"):
            with self.assertRaisesRegex(ValueError, "unexpected engine failure"):
                await self.invoke({"model": "decider"})

    def test_model_aliases_follow_standard_vllm_args(self):
        self.assertEqual(model_names(NS(served_model_name="alias"), NS(model="path")), ["alias"])
        self.assertEqual(model_names(NS(served_model_name=["one", "two"]), NS(model="path")), ["one", "two"])
        self.assertEqual(model_names(NS(), NS(model="path")), ["path"])


if __name__ == "__main__":
    unittest.main()
