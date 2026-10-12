# SPDX-License-Identifier: Apache-2.0
"""CPU test for SUFFIX_ROCM_GDN_MIXED_FAST (no vLLM, Triton or GPU): the step plan is built by
the first GDN layer of a step and reused by the others, a new step (metadata) builds a new
one, a declined step stays declined; _Launch replays the first call's compiled kernel with the
JITFunction.run argument list only while every tensor keeps its alignment; capture() takes
FLA's autotuned configs all or nothing. Silicon check: boot gate gdn_mixed_bench."""
import importlib.util
import sys
import types
from types import SimpleNamespace

import torch

if importlib.util.find_spec("vllm") is None:  # CPU CI: the module needs vllm.triton_utils only
    _tu = types.ModuleType("vllm.triton_utils")
    _tu.triton = SimpleNamespace(cdiv=lambda a, b: -(-a // b), next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    sys.modules.update({"vllm": types.ModuleType("vllm"), "vllm.triton_utils": _tu})
    try:
        from suffix_hybrid.kernels import gdn_mixed_fast_rocm as fm
    finally:
        del sys.modules["vllm"], sys.modules["vllm.triton_utils"]
from suffix_hybrid.kernels import gdn_mixed_fast_rocm as fm  # noqa: E402


def _step_args(md, plan):
    hv, dv, n, nst = 2, 4, 7, 3
    ssm = torch.zeros(5, hv, dv, dv)
    out = torch.zeros(9, hv, dv)
    layer = SimpleNamespace(conv1d=SimpleNamespace(bias=None), A_log=None, dt_bias=None)
    return (layer, torch.zeros(n, 20), torch.zeros(n, 2 * hv), out, md, plan, nst, n, 12, hv,
            torch.zeros(5, 12, 3), ssm, torch.zeros(12, 4))


def test_plan_built_once_per_step(monkeypatch):
    built, calls = [], []

    def build(*a):
        built.append(a)
        f = SimpleNamespace(h0=torch.zeros(1, 2, 4, 4), ht=torch.ones(1, 2, 4, 4),
                            conv=torch.zeros(4, 12), batch_ptr=None, tco_ptr=None, q=None, k=None,
                            v=None, g=None, beta=None)
        for name in ("conv", "post", "cumsum", "kkt", "merge", "wu", "h", "o"):
            setattr(f, f"L_{name}", lambda name=name, **kw: calls.append(name))
        return f

    monkeypatch.setattr(fm, "_build", build)
    monkeypatch.setattr(fm, "_CFG", {"cfg": {}})
    md = SimpleNamespace(non_spec_state_indices_tensor=None, has_initial_state=None,
                         non_spec_query_start_loc=None)
    plan = SimpleNamespace(pre_idx=torch.tensor([3]), no_init=torch.tensor([[[[False]]]]))
    for _ in range(3):  # three GDN layers of one step
        args = _step_args(md, plan)
        assert fm.prefill(*args)
        assert torch.equal(args[11][3], torch.ones(2, 4, 4))  # final state scattered back
        assert bool((args[3][7:] == 0).all())
    assert len(built) == 1 and calls.count("o") == 3 and calls[:8] == [
        "conv", "post", "cumsum", "kkt", "merge", "wu", "h", "o"]
    plan2 = SimpleNamespace(pre_idx=plan.pre_idx, no_init=plan.no_init)  # next step
    assert fm.prefill(*_step_args(md, plan2)) and len(built) == 2

    monkeypatch.setattr(fm, "_build", lambda *a: built.append(a) or False)
    plan3 = SimpleNamespace(pre_idx=plan.pre_idx, no_init=plan.no_init)
    assert not fm.prefill(*_step_args(md, plan3)) and not fm.prefill(*_step_args(md, plan3))
    assert len(built) == 3 and plan3.fast is False

    monkeypatch.setattr(fm, "_CFG", {})  # no autotuned configs yet: stock call first
    assert not fm.prefill(*_step_args(md, SimpleNamespace()))


class JITFunction:  # the name _inner walks to
    arg_names = ["x", "n", "BLOCK"]

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def run(**kw):
            self.launches.append((grid, kw))
            return self.compiled
        return run


def test_launch_replays_in_signature_order(monkeypatch):
    runs = []
    jit = JITFunction()
    jit.compiled = SimpleNamespace(
        function="fn", packed_metadata="meta", launch_metadata=lambda grid, stream, *a: None,
        run=lambda *a: runs.append(a))
    hooks = SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None)
    drv = SimpleNamespace(get_current_device=lambda: 0, get_current_stream=lambda d: "s0")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(knobs=SimpleNamespace(runtime=hooks)))
    monkeypatch.setitem(sys.modules, "triton.runtime", SimpleNamespace(driver=SimpleNamespace(active=drv)))
    monkeypatch.setattr(fm, "MODE", 2)
    launch = fm._Launch(SimpleNamespace(fn=jit), (4, 2), {"BLOCK": 64, "num_warps": 4})
    x = torch.zeros(64)
    launch(x=x, n=7)
    assert jit.launches == [((4, 2, 1), {"x": x, "n": 7, "BLOCK": 64, "num_warps": 4})]
    y = torch.zeros(64)
    launch(x=y, n=7)
    assert len(jit.launches) == 1 and runs == [(4, 2, 1, "s0", "fn", "meta", None, None, None, y, 7, 64)]
    launch(x=torch.zeros(65)[1:], n=7)  # a misaligned tensor: back through the JITFunction
    assert len(jit.launches) == 2 and len(runs) == 1
    monkeypatch.setattr(fm, "MODE", 1)
    launch(x=y, n=7)
    assert len(jit.launches) == 3 and len(runs) == 1


def test_capture_all_or_nothing(monkeypatch):
    class Autotuner:
        def __init__(self, cfg):
            self.best_config = cfg

    cfg = SimpleNamespace(all_kwargs=lambda: {"num_warps": 4})
    kernels = SimpleNamespace(**{n: SimpleNamespace(fn=Autotuner(cfg)) for n in fm.TUNED})
    monkeypatch.setattr(fm, "_kernels", lambda: kernels)
    monkeypatch.setattr(fm, "_CFG", {})
    kernels.o.fn.best_config = None  # one kernel not run yet
    fm.capture()
    assert fm._CFG == {}
    kernels.o.fn.best_config = cfg
    fm.capture()
    assert fm._CFG == {n: {"num_warps": 4} for n in fm.TUNED}
