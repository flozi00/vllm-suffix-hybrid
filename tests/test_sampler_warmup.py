# SPDX-License-Identifier: Apache-2.0
"""CPU wiring tests for suffix_hybrid.sampler_warmup (no vLLM, no GPU).

Silicon check (not testable here): boot a V2 pod with generation-config
top_k/top_p defaults, look for "[suffix sampler-warmup] warmed: ..." before
CUDA-graph capture, then send c=1..max_num_seqs requests and confirm the
vLLM jit_monitor logs NO "Triton kernel JIT compilation during inference:
_topk_topp_kernel/_topp_sb_*" line and the first-bench TTFT spike is gone.
"""
import sys
import types

from suffix_hybrid import sampler_warmup as sw


class _Kernel:
    def __init__(self, name, log, n=3):
        self.kernel = types.SimpleNamespace(__name__=name)
        self.log, self.n = log, n

    def get_warmup_keys(self, *, vllm_config):
        self.log.append(("keys", self.kernel.__name__, vllm_config))
        return list(range(self.n))

    def compile_many(self, keys):
        self.log.append(("compile", self.kernel.__name__, list(keys)))


def _runner(pooling=False, last=True):
    return types.SimpleNamespace(is_pooling_model=pooling, is_last_pp_rank=last,
                                 vllm_config="VC")


def _worker_mod(log):
    mod = types.ModuleType("fake_gpu_worker")

    def warmup_kernels(model_runner, execute, sample):
        log.append(("orig", execute, sample))
        return "ret"

    mod.warmup_kernels = warmup_kernels
    return mod


def test_warm_compiles_every_kernel_with_vllm_config(capsys):
    log = []
    ks = [_Kernel("_topk_topp_kernel", log), _Kernel("_topp_sb_mask_kernel", log, 2)]
    sw.warm(_runner(), kernels=ks)
    assert ("keys", "_topk_topp_kernel", "VC") in log
    assert ("compile", "_topp_sb_mask_kernel", [0, 1]) in log
    err = capsys.readouterr().err
    assert "[suffix sampler-warmup] warmed: _topk_topp_kernel,_topp_sb_mask_kernel (5 variants) in" in err


def test_warm_skips_pooling_and_non_last_pp_rank():
    log = []
    sw.warm(_runner(pooling=True), kernels=[_Kernel("k", log)])
    sw.warm(_runner(last=False), kernels=[_Kernel("k", log)])
    assert log == []


def test_patch_runs_after_orig_passes_result_and_is_idempotent(monkeypatch):
    log = []
    mod = _worker_mod(log)
    monkeypatch.setattr(sw, "warm", lambda mr: log.append(("warm", mr.vllm_config)))
    sw._patch(mod)
    wrapped = mod.warmup_kernels
    sw._patch(mod)
    assert mod.warmup_kernels is wrapped
    assert mod.warmup_kernels(_runner(), "ex", "st") == "ret"
    assert log == [("orig", "ex", "st"), ("warm", "VC")]


def test_patch_is_fail_soft(monkeypatch, capsys):
    mod = _worker_mod([])

    def boom(_):
        raise RuntimeError("triton says no")

    monkeypatch.setattr(sw, "warm", boom)
    sw._patch(mod)
    assert mod.warmup_kernels(_runner(), None, None) == "ret"
    assert "FAILED (serving unaffected" in capsys.readouterr().err


def test_gate_default_on_and_zero_disables(monkeypatch):
    monkeypatch.delenv(sw.GATE, raising=False)
    assert sw.enabled()
    monkeypatch.setenv(sw.GATE, "0")
    assert not sw.enabled()
    before = list(sys.meta_path)
    sw.install_post_import_hook()
    assert sys.meta_path == before


def test_finder_patches_real_import_once(monkeypatch, tmp_path):
    (tmp_path / "fake_vllm_gpu_worker.py").write_text(
        "def warmup_kernels(mr, ex, st):\n    return 'real'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sw, "TARGET", "fake_vllm_gpu_worker")
    monkeypatch.delenv(sw.GATE, raising=False)
    calls = []
    monkeypatch.setattr(sw, "warm", lambda mr: calls.append(mr))
    try:
        sw.install_post_import_hook()
        sw.install_post_import_hook()  # second arm is a no-op
        assert sum(getattr(f, sw._MARK, False) for f in sys.meta_path) == 1
        import fake_vllm_gpu_worker as m
        assert not any(getattr(f, sw._MARK, False) for f in sys.meta_path)
        assert getattr(m.warmup_kernels, sw._MARK)
        assert m.warmup_kernels("R", None, None) == "real" and calls == ["R"]
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, sw._MARK, False)]
        sys.modules.pop("fake_vllm_gpu_worker", None)
