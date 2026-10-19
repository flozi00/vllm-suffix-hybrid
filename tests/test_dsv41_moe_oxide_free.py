# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 is oxide-free (owner direction): nothing on its route imports
`suffix_hybrid._native` or `suffix_hybrid.oxide_kernels`, and the oxide / auto backends are refused
for V4 geometry with a message that says what to set. The route is exercised END TO END on CPU
under an import guard that raises on either module: arm() -> the layer class -> weight-processing
hook (eligibility, V4 gate, served-M table, workspaces, the layer oracle with its Helion warm-up)
-> the fused shared expert (prepare_layer: oracle, capture-replay, long-pole gate, runner detach)
-> forward at fused / split / vLLM-path M. vLLM is faked (RoutedExperts, MoERunner,
SharedExperts); the kernels are the real Helion bodies in eager reference mode."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")  # before Helion is imported

import ast  # noqa: E402
import importlib.abc  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from test_dsv41_moe_shared import (_every_config, _FakeShared, _Mlp,  # noqa: E402,F401
                                   _stock_mlp, helion)

import suffix_hybrid  # noqa: E402
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs  # noqa: E402
from suffix_hybrid.helion_kernels import nvfp4_moe as hm  # noqa: E402
from suffix_hybrid.kernels import dsv41_moe as dm  # noqa: E402
from suffix_hybrid.kernels import dsv41_moe_shared as glue  # noqa: E402
from suffix_hybrid.kernels import nvfp4_moe as nm  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1] / "suffix_hybrid"
FORBIDDEN = ("suffix_hybrid.oxide_kernels", "suffix_hybrid._native")


class _Forbid(importlib.abc.MetaPathFinder):
    """Raises on any import of the oxide family and records the attempt (a swallowed ImportError
    still fails the test through `hits`)."""

    def __init__(self):
        self.hits = []

    def find_spec(self, fullname, path, target=None):
        if fullname in FORBIDDEN or fullname.startswith("suffix_hybrid._native."):
            self.hits.append(fullname)
            raise ImportError(f"{fullname}: the DeepSeek-V4 route is oxide-free")
        return None


@pytest.fixture
def no_oxide(monkeypatch):
    finder = _Forbid()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for name in FORBIDDEN:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(suffix_hybrid, "oxide_kernels", raising=False)
    monkeypatch.delattr(suffix_hybrid, "_native", raising=False)
    yield finder
    assert finder.hits == [], f"the oxide-free route imported {finder.hits}"


# ---------------------------------------------------------------------------
# fake vLLM: RoutedExperts, MoERunner (the real hook wraps its __init__), SharedExperts
# ---------------------------------------------------------------------------
HDIM, IDIM, ISZ, LOCAL, EG, BASE, TOPK, LIMIT = 128, 288, 64, 6, 48, 24, 6, 10.0


class _Base:
    """vLLM's RoutedExperts as the lane reads it; forward_modular = the FlashInfer path (the f64
    spec with FlashInfer's bf16 intermediates)."""
    registered = []
    parent_calls = []

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def register_oot(cls, oot, name=None):
        _Base.registered.append(oot)

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                        shared_experts_input=None):
        _Base.parent_calls.append(x.shape[0])
        lid = nm.to_local(topk_ids, BASE, LOCAL)
        return nm.moe_ref(self._p, x, lid, topk_weights, fi=True).bfloat16()


class _Runner:
    """vLLM's MoERunner (__init__ signature = the anchors install_runner_hook checks)."""

    def __init__(self, layer_name, moe_config, router, routed_experts, enable_dbo=False, gate=None,
                 shared_experts=None, shared_expert_gate=None, routed_input_transform=None,
                 routed_output_transform=None, routed_scaling_factor=1.0):
        self.layer_name, self.moe_config = layer_name, moe_config
        self.routed_experts, self._shared_experts = routed_experts, shared_experts
        self.shared_expert_gate = shared_expert_gate
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor
        self._fused_output_is_reduced = False
        self._forward_entry = self._select_forward()

    def _select_forward(self):
        return "moe_forward" if self._shared_experts is None else "moe_forward_shared"


def _install_vllm(monkeypatch):
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe",
             "vllm.model_executor.layers.fused_moe.routed_experts",
             "vllm.model_executor.layers.fused_moe.runner",
             "vllm.model_executor.layers.fused_moe.runner.moe_runner",
             "vllm.model_executor.layers.fused_moe.runner.shared_experts"]
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    sys.modules[names[4]].RoutedExperts = _Base
    sys.modules[names[6]].MoERunner = _Runner
    sys.modules[names[7]].SharedExpertsOrder = types.SimpleNamespace(NO_OVERLAP=1)
    monkeypatch.setattr(_Runner, "__init__", _Runner.__init__)  # restored: the hook patches it
    monkeypatch.setattr(_Base, "registered", [])
    monkeypatch.setattr(_Base, "parent_calls", [])


def _env(monkeypatch, backend="helion", fuse=True):
    for k, v in (("SUFFIX_NVFP4_MOE", "1"), ("SUFFIX_HELION_NVFP4_MOE", backend),
                 ("SUFFIX_NVFP4_MOE_MAX_M", "8"), ("SUFFIX_DSV41_MOE_FUSE_SHARED", "1" if fuse else "0"),
                 ("SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M", "4")):
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("SUFFIX_FP8_MOE", raising=False)
    # the V4.1 gate at the test's small dims (the checks themselves are pinned in test_dsv41_refuse)
    monkeypatch.setattr(dm, "HIDDEN", HDIM)
    monkeypatch.setattr(dm, "E_GLOBAL", EG)
    monkeypatch.setattr(dm, "LAYOUTS", {(LOCAL, IDIM): ("EP8", 8)})
    monkeypatch.setattr(hk, "config", _every_config)
    monkeypatch.setattr(hk, "platform", lambda device=0: "test_gpu")
    hm._kernels.cache_clear()
    hs._kernels_sh.cache_clear()
    for key, fresh in (("armed", False), ("instances", 0), ("layers_ours", 0), ("stock", {}),
                       ("ws", {}), ("checked", False), ("hook", None), ("oracle", []),
                       ("active_logged", False), ("helion_warm", set()), ("in_oracle", False),
                       ("shared_fused", 0)):
        monkeypatch.setitem(nm._state, key, fresh)


def _layer(cls, p, backend="helion"):
    qc = types.SimpleNamespace(
        quant_dtype="nvfp4", is_scale_swizzled=True, gemm1_clamp_limit=LIMIT, gemm1_alpha=None,
        w1_bias=None, w1_scale=p["w13_sf"], w2_scale=p["w2_sf"], g1_alphas=p["g1"],
        g2_alphas=p["g2"], a1_gscale=torch.full((LOCAL,), p["a1g"]),
        a2_gscale=torch.full((LOCAL,), p["a2g"]))
    qm = types.SimpleNamespace(
        moe_quant_config=qc, nvfp4_backend=types.SimpleNamespace(name="FLASHINFER_CUTLASS"),
        mk_can_overlap_shared_experts=False, process_weights_after_loading=lambda layer: None)
    pc = types.SimpleNamespace(use_all2all_kernels=False, enable_eplb=False, ep_rank=BASE // LOCAL)
    mc = types.SimpleNamespace(tp_size=1, ep_size=EG // LOCAL, dp_size=1, moe_parallel_config=pc,
                               is_sequence_parallel=False)
    emap = torch.tensor([g - BASE if BASE <= g < BASE + LOCAL else -1 for g in range(EG)])
    return cls(quant_method=qm, moe_config=mc, activation="silu", expert_map=emap,
               apply_router_weight_on_input=False, global_num_experts=EG, top_k=TOPK,
               layer_name="model.layers.3.ffn.experts.routed_experts",
               w13_weight=p["w13"], w2_weight=p["w2"], _p=p)


def _arm(monkeypatch):
    """nm.arm() on a host with no GPU: only the device probes and the global hook are stubbed."""
    with monkeypatch.context() as m:
        m.setattr(torch.cuda, "is_available", lambda: True)
        m.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
        m.setattr(torch.nn.modules.module, "register_module_forward_pre_hook",
                  lambda fn: types.SimpleNamespace(remove=lambda: None))
        nm.arm()
    return _Base.registered[-1]


def _cuda_stubs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_stream", lambda dev=None: types.SimpleNamespace(
        cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda dev=None: None)


def _timer(monkeypatch):
    monkeypatch.setattr(nm, "_graph_us", lambda fn, dev, iters, flush=None: (fn(None), 10.0)[1])


# ---------------------------------------------------------------------------
# the end-to-end route
# ---------------------------------------------------------------------------
def test_v41_route_end_to_end_imports_no_oxide(helion, monkeypatch, no_oxide):  # noqa: F811
    _install_vllm(monkeypatch)
    _env(monkeypatch)
    _cuda_stubs(monkeypatch)
    _timer(monkeypatch)
    assert nm.oxide_free_route() is True
    cls = _arm(monkeypatch)                      # arm(): no manifest, no cubin, no native check
    assert nm._state["armed"] and cls.__name__ == "SuffixRoutedExperts"
    p = nm.make_problem(LOCAL, HDIM, IDIM, "silu", seed=11, limit=LIMIT)
    sp = hs.make_shared_problem(HDIM, ISZ, seed=3, limit=LIMIT, wstd=0.05)
    layer = _layer(cls, p)
    mlp = _Mlp(sp, "FlashInferCutlassMxfp8LinearKernel", LIMIT, _stock_mlp(sp))
    runner = _Runner(layer.layer_name, types.SimpleNamespace(is_sequence_parallel=False), None,
                     layer, shared_experts=_FakeShared(mlp, True))
    assert layer._sfx_runner() is runner         # the hooked MoERunner.__init__ wired the layer
    layer.quant_method.process_weights_after_loading(layer)   # eligibility .. oracle .. fusion
    cfg = layer._sfx_moe
    assert nm._state["layers_ours"] == 1 and nm._state["ws"][layer.w13_weight.device] is nm.NO_WS
    assert runner._shared_experts is None and runner._forward_entry == "moe_forward"
    assert cfg["op"].backend == "helion" and "dsv41_shared" in cfg
    assert all(v[2] == "dequant" for v in cfg["helion_ms"].values())
    # forward at a fused M, a split M (routed on Helion + the module) and a vLLM-path M
    make = glue.batch_factory(cfg, HDIM, torch.device("cpu"))
    for m, via in ((2, "fused"), (6, "split"), (9, "vllm")):
        x, ids, tw = make(m, 40 + m, "balanced")
        out = layer.forward_modular(x, tw, ids)
        routed = nm.moe_ref(cfg, x, nm.to_local(ids, BASE, LOCAL), tw)
        if via == "fused":      # our shared slot: the exact f64 spec of routed + shared
            assert hm._rel(out.float(), routed + hs.shared_ref(sp, x)) <= nm.REF_TOL
        else:                   # vLLM's module (MXFP8 activations) + routed on Helion / FlashInfer
            tol = nm.REF_TOL if via == "split" else glue.FULL_STOCK_TOL
            assert hm._rel(out.float(), routed + mlp(x).double()) <= tol, (m, via)
    assert _Base.parent_calls[-1] == 9
    assert no_oxide.hits == []


def test_v41_route_without_the_shared_fusion_is_oxide_free_too(helion, monkeypatch,  # noqa: F811
                                                               no_oxide):
    _install_vllm(monkeypatch)
    _env(monkeypatch, fuse=False)
    _cuda_stubs(monkeypatch)
    cls = _arm(monkeypatch)
    p = nm.make_problem(LOCAL, HDIM, IDIM, "silu", seed=11, limit=LIMIT)
    layer = _layer(cls, p)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer._sfx_moe is not None and "dsv41_shared" not in layer._sfx_moe
    x, ids, tw = glue.batch_factory(layer._sfx_moe, HDIM, torch.device("cpu"))(5, 3, "skewed")
    ref = nm.moe_ref(layer._sfx_moe, x, nm.to_local(ids, BASE, LOCAL), tw)
    assert hm._rel(layer.forward_modular(x, tw, ids).float(), ref) <= nm.REF_TOL


@pytest.mark.parametrize("backend", ["oxide", "auto"])
def test_oxide_and_auto_backends_are_refused_for_v41_geometry(helion, monkeypatch,  # noqa: F811
                                                              no_oxide, backend):
    _install_vllm(monkeypatch)
    _env(monkeypatch, backend=backend, fuse=False)
    p = nm.make_problem(LOCAL, HDIM, IDIM, "silu", seed=11, limit=LIMIT)
    layer = _layer(nm._make_layer_cls(None), p)
    with pytest.raises(RuntimeError) as exc:
        layer.quant_method.process_weights_after_loading(layer)
    msg = str(exc.value)
    assert "DeepSeek-V4 NVFP4 MoE layer model.layers.3.ffn.experts.routed_experts" in msg
    assert f"SUFFIX_HELION_NVFP4_MOE={backend!r}" in msg and "oxide-free" in msg
    assert "swiglu clamp 10.0" in msg and "set SUFFIX_HELION_NVFP4_MOE=helion" in msg
    assert layer._sfx_moe is None and nm._state["layers_ours"] == 0


def test_backend_refusal_text_and_the_oxide_free_predicate(monkeypatch):
    assert dm.backend_refusal("helion", "E", "gap") is None
    assert "set E=helion" in dm.backend_refusal("auto", "E", None)
    monkeypatch.setenv("SUFFIX_NVFP4_MOE", "1")
    monkeypatch.delenv("SUFFIX_FP8_MOE", raising=False)
    for backend, free in (("helion", True), ("auto", False), ("oxide", False)):
        monkeypatch.setenv("SUFFIX_HELION_NVFP4_MOE", backend)
        assert nm.oxide_free_route() is free
    monkeypatch.setenv("SUFFIX_HELION_NVFP4_MOE", "helion")
    monkeypatch.setenv("SUFFIX_FP8_MOE", "1")   # an FP8 block layer needs its oxide family
    assert nm.oxide_free_route() is False
    monkeypatch.delenv("SUFFIX_FP8_MOE")
    monkeypatch.setenv("SUFFIX_NVFP4_MOE", "0")
    assert nm.oxide_free_route() is False


def test_runner_hook_refuses_a_drifted_moe_runner(monkeypatch):
    _install_vllm(monkeypatch)
    cls = type("Layer", (), {})
    glue.install_runner_hook(cls)
    layer = cls()
    r = _Runner("l", None, None, layer)
    assert layer._sfx_runner() is r and _Runner.__init__._sfx_hooked
    first = _Runner.__init__
    glue.install_runner_hook(cls)               # idempotent: no double wrap
    assert _Runner.__init__ is first and first.__wrapped__ is not first

    class Drifted:
        def __init__(self, layer_name, routed_experts):
            pass

    sys.modules["vllm.model_executor.layers.fused_moe.runner.moe_runner"].MoERunner = Drifted
    with pytest.raises(RuntimeError, match=r"MoERunner drifted.*shared_expert_gate"):
        glue.install_runner_hook(cls)


# ---------------------------------------------------------------------------
# where oxide may be imported at all (the source-level guard)
# ---------------------------------------------------------------------------
def _oxide_importers(path):
    """{enclosing function: [imported names]} for every import of oxide_kernels / _native."""
    tree = ast.parse(path.read_text())
    found = {}

    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            sc = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in child.names] + [getattr(child, "module", None) or ""]
                if any("oxide_kernels" in n or n == "_native" or "._native" in n for n in names):
                    found.setdefault(scope, []).append(ast.unparse(child))
            walk(child, sc)
    walk(tree, "<module>")
    return found


def test_oxide_is_imported_only_where_the_oxide_backends_live():
    """The DeepSeek-V4 route's modules never import it; nvfp4_moe may only inside arm (guarded
    by `not oxide_free`), the layer class's guarded ensure_loaded, the oxide-only standalone
    helpers and the oxide lab baselines. A new import elsewhere fails here."""
    never = ("kernels/dsv41_moe.py", "kernels/dsv41_moe_shared.py",
             "helion_kernels/dsv41_moe_shared.py", "helion_kernels/nvfp4_moe_tune.py")
    for rel in never:
        assert _oxide_importers(ROOT / rel) == {}, rel
    assert set(_oxide_importers(ROOT / "kernels/nvfp4_moe.py")) == {"arm", "_sfx_prepare",
                                                                    "_native_ready"}
    assert set(_oxide_importers(ROOT / "helion_kernels/nvfp4_moe.py")) == {"_oxide_args",
                                                                           "_oxide_baseline"}
    src = (ROOT / "kernels/nvfp4_moe.py").read_text()
    # arm: the import sits in the `if not oxide_free` branch; prepare: behind the helion/gap test
    assert "    if not oxide_free:\n        from suffix_hybrid import oxide_kernels" in src
    assert 'if hb != "helion" and not gap:' in src


def test_v41_standalone_gates_only_run_helion_only_cases():
    """`oracle_v41` / `bench_v41` take no native: every case in them is one `_dev_problem` serves
    with a Helion op (oxide_gap non-empty), and the CLI knows the modes."""
    for cases in (nm.V41_ORACLE, nm.V41_BENCH):
        assert [c[0] for c in cases] == [nm.DS_EP, nm.DS_TP8]
        for case, _ in cases:
            e_global, hdim, idim, topk, act, ep_size, _r = case[:7]
            assert nm.oxide_gap({"clamp_limit": case[8], "E": e_global // ep_size, "I": idim})
    src = (ROOT / "kernels/nvfp4_moe.py").read_text()
    assert 'mode == "oracle_v41"' in src and 'mode == "bench_v41"' in src
