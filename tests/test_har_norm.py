import sys
import types

import torch

from suffix_hybrid import har_norm, host_ar


class RMSNorm(torch.nn.Module):  # the name the static check looks at
    def __init__(self, h):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(h, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-5
        self.has_weight = True


def test_routes_plain_bf16_rmsnorm_to_the_op_and_falls_back_otherwise(monkeypatch):
    calls = []

    def orig(h, r, norm):
        calls.append("orig")
        return h, r

    mod = types.SimpleNamespace(fused_allreduce_rms_norm=orig)
    monkeypatch.setattr(har_norm, "_register_op", lambda: None)
    fake_ops = types.SimpleNamespace(ar_rmsnorm=lambda x, r, w, e: (calls.append(("op", e)) or (x, r)))
    monkeypatch.setattr(torch.ops, "suffix_har", fake_ops, raising=False)
    monkeypatch.setattr(host_ar, "TP_COMM", object())
    har_norm.apply(mod)
    x = torch.zeros(6, 64, dtype=torch.bfloat16)
    mod.fused_allreduce_rms_norm(x, x.clone(), RMSNorm(64))
    mod.fused_allreduce_rms_norm(x.float(), x.float(), RMSNorm(64))  # fp32 -> upstream path
    monkeypatch.setattr(host_ar, "TP_COMM", None)
    mod.fused_allreduce_rms_norm(x, x.clone(), RMSNorm(64))  # no host buffer -> upstream path
    assert calls == [("op", 1e-5), "orig", "orig"]
