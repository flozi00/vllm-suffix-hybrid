"""Contract tests with real tensors; vLLM/CUDA execution is a deployment gate."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from suffix_hybrid import wrap


def runner():
    return NS(
        use_async_scheduling=False,
        speculative_config=NS(method="mtp", disable_padded_drafter_batch=True),
        parallel_config=NS(pipeline_parallel_size=1),
        model_config=NS(is_hybrid=False),
        num_spec_tokens=4, max_model_len=32,
        input_batch=NS(req_ids=["a", "b"], num_tokens_no_spec=np.array([3, 2]),
                       token_ids_cpu=np.array([[1, 2, 3, 0], [7, 8, 0, 0]])),
        rejection_sampler=NS(synthetic_mode=False),
        drafter=type("EagleProposer", (), {"__module__": "vllm.v1.spec_decode.eagle"})(),
    )


def invoke(fn, r, sampled=None, metadata=None, greedy=True):
    return fn(r, NS(num_spec_tokens_to_schedule=4), sampled or [[3], [8]],
              NS(all_greedy=greedy), None, None, None, metadata, None, None)


def test_native_runs_first_and_rust_mixes_with_authoritative_context():
    r = runner()
    calls = []
    native = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32)

    def propose(self, *args):
        calls.append("native")
        self._draft_probs = None
        self._draft_prob_req_ids = None
        return native

    class Mixer:
        def __init__(self, k, max_len):
            assert (k, max_len) == (4, 32)

        def mix_numpy(self, ids, counts, tokens, drafts, accepted, step_time_ns):
            calls.append("rust")
            assert ids == ["a", "b"]
            assert counts is r.input_batch.num_tokens_no_spec
            assert tokens is r.input_batch.token_ids_cpu
            assert drafts == native.tolist()
            assert accepted == [-1, -1]
            assert step_time_ns is None
            return [[10, 11, 12, 99], [20, 21, 22, 23]]

    result = invoke(wrap._wrap_propose(propose, Mixer), r)
    assert calls == ["native", "rust"]
    assert result == [[10, 11, 12, 99], [20, 21, 22, 23]]
    assert native.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23]]


@pytest.mark.parametrize("case", ["stochastic", "async", "padded", "pipeline", "recurrent", "synthetic", "wrong_method", "tensor_sampled"])
def test_unsupported_contract_rejected_before_native(case):
    r = runner()
    if case == "async": r.use_async_scheduling = True
    if case == "padded": r.speculative_config.disable_padded_drafter_batch = False
    if case == "pipeline": r.parallel_config.pipeline_parallel_size = 2
    if case == "recurrent": r.model_config.is_hybrid = True
    if case == "synthetic": r.rejection_sampler.synthetic_mode = True
    if case == "wrong_method": r.speculative_config.method = "custom_class"
    def native(*args):
        pytest.fail("must reject before executing native")
    fn = wrap._wrap_propose(native, None)
    with pytest.raises(RuntimeError, match="suffix hybrid"):
        if case == "tensor_sampled":
            fn(r, NS(), torch.tensor([[3], [8]]), NS(all_greedy=True), None, None, None, None, None, None)
        else:
            invoke(fn, r, greedy=case != "stochastic")


def test_feedback_uses_accepted_and_scheduled_counts_with_discard_censoring():
    r = runner()
    r.input_batch.req_ids = ["b", "a"]
    class Mixer:
        def __init__(self, *args): pass
        def mix_numpy(self, ids, counts, tokens, drafts, accepted, step_time_ns):
            assert ids == ["b", "a"]
            assert accepted == [1, -1]  # one accepted, plus bonus; discarded is unknown
            assert step_time_ns is None
            assert drafts[1] == []     # chunked prefill/discard must not receive drafts
            return drafts
    def native(self, *args):
        self._draft_probs = None
        return torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    output = invoke(wrap._wrap_propose(native, Mixer), r,
                    sampled=[[2, 3], []], metadata=NS(num_draft_tokens=[2, 3]))
    assert output[1] == []


def test_native_probabilities_preserved_suffix_tail_is_one_hot():
    r = runner()
    probabilities = torch.rand(2, 4, 100)
    original = probabilities.clone()
    class Mixer:
        def __init__(self, *args): pass
        def mix_numpy(self, *args): return [[10, 11, 12, 99], [20, 21, 22, 23]]
        def last_native_counts(self): return [3, 4]
    def native(self, *args):
        self._draft_probs = probabilities
        self._draft_prob_req_ids = ["a", "b"]
        return torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    invoke(wrap._wrap_propose(native, Mixer), r)
    assert torch.equal(r._draft_probs[0, :3], original[0, :3])
    assert torch.equal(r._draft_probs[1], original[1])
    assert r._draft_probs[0, 3, 99] == 1
    assert r._draft_probs[0, 3].sum() == 1
    assert torch.equal(probabilities, original)  # don't modify native-owned buffer
    assert r._draft_prob_req_ids == ["a", "b"]


def test_scheduler_truncation_censored_instead_of_false_rejection():
    r = runner()
    r._suffix_hybrid_lengths = {"a": 4, "b": 2}
    r._suffix_hybrid_cost_ns = 123
    class Mixer:
        def __init__(self, *args): pass
        def mix_numpy(self, ids, counts, tokens, drafts, accepted, cost):
            assert accepted == [-1, 2]  # a truncated; b actually fully accepted
            assert cost == 123
            return [[1, 2], [3]]
    def native(self, *args):
        self._draft_probs = None
        return torch.ones((2, 4), dtype=torch.int32)
    invoke(wrap._wrap_propose(native, Mixer), r, sampled=[[1, 2, 3], [4, 5, 6]],
           metadata=NS(num_draft_tokens=[2, 2]))
    assert r._suffix_hybrid_lengths == {"a": 2, "b": 1}
    assert r._suffix_hybrid_cost_ns > 0


def test_qwen35_full_attention_mtp_allowed_with_hybrid_target():
    r = runner()
    r.model_config.is_hybrid = True
    r.speculative_config.draft_model_config = NS(hf_config=NS(model_type="qwen3_5_mtp"))
    class Mixer:
        def __init__(self, *args): pass
        def mix_numpy(self, ids, counts, tokens, drafts, *args): return drafts
    def native(self, *args):
        self._draft_probs = None
        return torch.ones((2, 4), dtype=torch.int32)
    assert invoke(wrap._wrap_propose(native, Mixer), r) == [[1]*4, [1]*4]


def test_exact_source_gate_rejects_changed_file(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(wrap, "_SOURCE_HASHES", {"runner.py": hashlib.sha256(b"pinned").hexdigest()})
    path = tmp_path / "runner.py"
    path.write_bytes(b"pinned")
    wrap._verify_sources(tmp_path)
    path.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="d05da62e9"):
        wrap._verify_sources(tmp_path)


def test_install_patches_runner_once_not_drafter(monkeypatch, tmp_path):
    import sys
    import suffix_hybrid._native as rust
    class Runner:
        def propose_draft_token_ids(self, *args): return torch.zeros((2, 4), dtype=torch.int32)
    original = Runner.propose_draft_token_ids
    monkeypatch.setenv("SUFFIX_HYBRID_WRAP", "1")
    monkeypatch.setitem(sys.modules, "vllm", NS(__file__=str(tmp_path / "__init__.py")))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", NS(GPUModelRunner=Runner))
    monkeypatch.setattr(wrap, "_verify_sources", lambda root: None)
    monkeypatch.setattr(rust, "HybridMixer", object, raising=False)
    assert wrap.install() is True
    first = Runner.propose_draft_token_ids
    assert first is not original
    assert first.__wrapped__ is original
    assert wrap.install() is True
    assert Runner.propose_draft_token_ids is first


def test_real_rust_mixer_keeps_native_prefix_and_conditions_suffix(monkeypatch, capsys):
    from suffix_hybrid._native import HybridMixer
    monkeypatch.setenv("SUFFIX_HYBRID_INDEX_N", "2")
    monkeypatch.setenv("SUFFIX_HYBRID_LOG_INTERVAL", "1")
    r = runner()
    r._suffix_hybrid_mixer = HybridMixer(4, 32)
    r._suffix_hybrid_mixer.suffix_cache.add_sequence([1, 2, 3, 10, 11, 12, 99])
    calls = []
    def native(self, *args):
        calls.append(True)
        self._draft_probs = None
        return torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    fn = wrap._wrap_propose(native, HybridMixer)
    result = invoke(fn, r)
    assert calls == [True]
    # Fresh rows publish native-only (no per-row evidence yet); the second
    # proposal carries the learned split: cache tail replaces the 4th slot.
    assert result == [[10, 11, 12, 13], [20, 21, 22, 23]]
    # Second proposal: the row is now seen, gate open (no tail feedback
    # yet -> one trial). n=2 lookup on [1,2,3,10,11] yields [12,99],
    # replacing the unearned 4th native slot. suffix_proposed counts the
    # one genuinely non-native slot.
    result2 = invoke(fn, r)
    assert result2[0] == [10, 11, 12, 99]
    assert r._suffix_hybrid_mixer.get_stats()["suffix_proposed"] == 1
    assert '"suffix_proposed": 1' in capsys.readouterr().err


def test_sitecustomize_enabled_failure_is_fatal(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    package = tmp_path / "suffix_hybrid"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "wrap.py").write_text("def install(): raise RuntimeError('contract mismatch')\n")
    startup = Path(wrap.__file__).parents[1] / "sitecustomize.py"
    (tmp_path / "sitecustomize.py").write_bytes(startup.read_bytes())
    env = dict(os.environ, SUFFIX_HYBRID_WRAP="1", PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", "print('silently continued')"],
                            env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "silently continued" not in result.stdout


def test_native_list_not_mutated_and_short_mixed_lists_not_padded():
    r = runner()
    native_rows = [[10, 11, 12, 13], [20, 21, 22, 23]]
    class Mixer:
        def __init__(self, *args): pass
        def mix_numpy(self, ids, counts, tokens, drafts, *args):
            assert drafts == [[10, 11, 12, 13], []]
            return [[10, 11], []]
    def native(self, *args):
        self._draft_probs = None
        return native_rows
    assert invoke(wrap._wrap_propose(native, Mixer), r, sampled=[[3], []]) == [[10, 11], []]
    assert native_rows[1] == [20, 21, 22, 23]


def test_unaudited_drafter_subclass_rejected():
    r = runner()
    r.drafter = object()
    with pytest.raises(RuntimeError, match="drafter"):
        invoke(wrap._wrap_propose(lambda *args: pytest.fail("must not run"), None), r)
