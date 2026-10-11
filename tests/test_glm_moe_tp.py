import dataclasses
import types

import torch

from suffix_hybrid.kernels import nvfp4_moe
from suffix_hybrid.tools.glm_moe_tp import worst_rank_ids


def test_worst_rank_lands_on_the_measured_rank():
    # rank 2 (experts 64..95) gets 3 distinct experts, everybody else at most 1
    ids = torch.tensor([[64, 65, 66, 0, 40, 100, 130, 200]])
    rot, counts = worst_rank_ids(ids)
    assert counts[2] == 3 and max(c for r, c in enumerate(counts) if r != 2) <= 1
    assert sorted(int(i) for i in rot[0] if 160 <= i < 192) == [160, 161, 162]
    assert torch.equal((rot - ids) % 256, torch.full_like(ids, 96))  # one rotation for every id


def test_scope_backend_only_rewrites_auto(monkeypatch):
    @dataclasses.dataclass
    class Cfg:
        moe_backend: str = "auto"

    mod = types.SimpleNamespace(select_nvfp4_moe_backend=lambda config, **kw: config.moe_backend)
    monkeypatch.setenv("SUFFIX_NVFP4_MOE_BACKEND", "flashinfer_b12x")
    nvfp4_moe.scope_backend(mod)
    assert mod.select_nvfp4_moe_backend(config=Cfg()) == "flashinfer_b12x"
    assert mod.select_nvfp4_moe_backend(config=Cfg("marlin")) == "marlin"  # explicit flag wins
