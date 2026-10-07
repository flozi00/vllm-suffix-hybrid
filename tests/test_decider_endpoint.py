"""Exercise native endpoint boundary without requiring a GPU or vLLM install."""
import json
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import suffix_hybrid
from suffix_hybrid_decider_ep import DecisionEndpointPlugin, model_names


class Response:
    def __init__(self, content, status_code=200):
        self.content, self.status_code = content, status_code


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
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
        self.modules = {"fastapi": fastapi, "fastapi.responses": responses,
                        "vllm.pooling_params": pooling}
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

    def test_model_aliases_follow_standard_vllm_args(self):
        self.assertEqual(model_names(NS(served_model_name="alias"), NS(model="path")), ["alias"])
        self.assertEqual(model_names(NS(served_model_name=["one", "two"]), NS(model="path")), ["one", "two"])
        self.assertEqual(model_names(NS(), NS(model="path")), ["path"])


if __name__ == "__main__":
    unittest.main()
