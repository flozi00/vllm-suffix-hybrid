# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sm120/hisparse_mtp_patch: fixture replay, gate-off
inertness, drift -> PatchDriftError, functional semantics of the rewritten
builder logic on a synthetic module, hook arming, and the oracle verdict."""
import hashlib
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))
import hisparse_mtp_patch as P  # noqa: E402
from hisparse_mtp_patch import oracle  # noqa: E402
from nvfp4_kv_patch import _PostImportFinder  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures/vllm_0.30.0/sparse_mla_attention.py"
FIXTURE_SHA256 = "ae0aa4fee580a5fcb22f912078d2df142dd7e576c790383211e5c06e484c0e27"
FIXTURES = FIXTURE.parent
RUNTIME_FIXTURE = FIXTURES / "hisparse_runtime.py"  # vllm/v1/hisparse/runtime.py
RUNTIME_SHA256 = "84a5305c3aa4a047c8ecb2906f6354b0d8bfe3dcd40e34f5a7422ddc15f56344"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(P.GATE_ENV, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


def test_fixture_is_pinned_wheel_copy():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert hashlib.sha256(RUNTIME_FIXTURE.read_bytes()).hexdigest() == RUNTIME_SHA256


def _defs(path, *names, ns=None):
    """exec the named top-level functions / class methods of a fixture."""
    import ast
    tree = ast.parse(path.read_text())
    ns = {} if ns is None else ns
    ns.setdefault("VllmConfig", object)  # annotation (runtime.py has no pep563 here)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns


def _cfg(k, seqs=1, mbt=2048):
    spec = None if k is None else types.SimpleNamespace(
        num_speculative_tokens=k, parallel_drafting=False)
    return types.SimpleNamespace(
        speculative_config=spec,
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=seqs, max_num_batched_tokens=mbt))


def _step_rows(capacity):
    ns = _defs(RUNTIME_FIXTURE, "_step_rows")
    rt = types.SimpleNamespace(_swap_step=0, index_group=types.SimpleNamespace(
        shared_topk=types.SimpleNamespace(
            physical_topk_indices=types.SimpleNamespace(shape=(capacity, 2048)))))
    return lambda n: ns["_step_rows"](rt, n)


def test_silicon_capacity_error_replays_on_cpu_and_patch_fixes_it():
    """k=3, max_num_seqs=1 (the oracle): index-group workspace has 4+1 rows
    and admits a 5-token all-resident prefill into swap_in; stock swap state
    has 4 rows -> the exact silicon ValueError. Patched: 5 rows, fits."""
    ig = (FIXTURES / "index_group.py").read_text()
    assert ig.count("        if num_tokens > self.physical_topk_indices.shape[0]:\n"
                    "            # Prefill-sized batches do not fit") == 1
    assert ig.count("(workspace_rows + 1, self.logical_topk_indices.shape[1])") == 1
    group_rows = _defs(FIXTURES / "index_group.py",
                       "get_sparse_mla_index_group_max_rows")[
        "get_sparse_mla_index_group_max_rows"]
    stock = _defs(RUNTIME_FIXTURE, "_get_max_decode_query_len",
                  "_get_max_swap_rows")
    patched = dict(stock)
    exec(P.patch_runtime_source(RUNTIME_FIXTURE.read_text()), patched)
    cfg = _cfg(3)
    workspace = group_rows(cfg) + 1
    assert (workspace, stock["_get_max_swap_rows"](cfg)) == (5, 4)
    n = workspace  # largest batch index_group.py:236 routes to swap_in
    with pytest.raises(ValueError, match=r"stop=5, capacity=4\.$"):
        _step_rows(stock["_get_max_swap_rows"](cfg))(n)
    assert _step_rows(patched["_get_max_swap_rows"](cfg))(n) == slice(0, 5)
    # Multi-token decode (patched SM120 path): one swap per verify step.
    step = _step_rows(patched["_get_max_swap_rows"](cfg))
    assert [step(1) for _ in range(4)][-1] == slice(3, 4)
    # Invariant for every shape: swap rows == index-group workspace rows.
    for k in (None, 1, 3, 5):
        for seqs in (1, 4, 64):
            for mbt in (8, 2048):
                c = _cfg(k, seqs, mbt)
                assert patched["_get_max_swap_rows"](c) == group_rows(c) + 1
                assert stock["_get_max_swap_rows"](c) == group_rows(c)


def test_runtime_anchor_drift():
    src = RUNTIME_FIXTURE.read_text()
    assert P.HELPER_TAG in P.patch_runtime_source(src)
    with pytest.raises(P.PatchDriftError, match="swap_rows_def: expected 1, found 0"):
        P.patch_runtime_source(src.replace(P.RUNTIME_OLD, P.RUNTIME_NEW))
    with pytest.raises(P.PatchDriftError, match="swap_rows_call: expected 1, found 2"):
        P.patch_runtime_source(src + P.RUNTIME_CALL_SITE)
    with pytest.raises(P.PatchDriftError, match="on disk"):
        P.patch_runtime_source(src + "# " + P.HELPER_TAG)


def test_replay_on_fixture():
    src = FIXTURE.read_text()
    new, applied = P.patch_source(src)
    assert applied == ["helper_block", "hisparse_spec_as_decode", "build_marker"]
    for _name, old, rep, _c in P.EDITS:
        assert new.count(rep) == 1
    assert "if not self.hisparse_supports_multi_token_decode:" not in new
    assert new.count(P.HELPER_TAG) == 2
    # Byte-exact inverse: nothing outside the three hunks changed.
    back = new
    for _name, old, rep, _c in reversed(P.EDITS):
        back = back.replace(rep, old)
    assert back == src
    compile(new, "x", "exec")


def test_replay_twice_is_drift():
    new, _ = P.patch_source(FIXTURE.read_text())
    with pytest.raises(P.PatchDriftError, match="hisparse_spec_as_decode"):
        P.patch_source(new)


@pytest.mark.parametrize("name", [e[0] for e in P.EDITS])
def test_drift_missing_and_duplicate_anchor(name):
    src = FIXTURE.read_text()
    old = next(e[1] for e in P.EDITS if e[0] == name)
    with pytest.raises(P.PatchDriftError, match=f"{name}: expected 1, found 0"):
        P.patch_source(src.replace(old, old.replace("=", "= ", 1)))
    with pytest.raises(P.PatchDriftError, match=f"{name}: expected 1, found 2"):
        P.patch_source(src + "\n" + old)


def test_gate_off_is_inert(monkeypatch):
    before = list(sys.meta_path)
    mod = types.ModuleType("m")
    mod.__file__ = str(FIXTURE)
    snapshot = dict(mod.__dict__)
    assert P.install_post_import_hook() is False
    assert P.apply(mod) is False
    assert sys.meta_path == before and mod.__dict__ == snapshot


def test_gate_on_not_sm120_is_inert(monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: False)
    mod = types.ModuleType("m")
    assert P.apply(mod) is False and not hasattr(mod, P.MARKER_ATTR)


def test_hook_arms_at_front(monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.delitem(sys.modules, P.TARGET_MODULE, raising=False)
    assert P.install_post_import_hook() is True
    f = sys.meta_path[0]
    assert isinstance(f, _PostImportFinder) and f.target == P.TARGET_MODULE
    assert P.install_post_import_hook() is True  # idempotent
    assert sum(isinstance(x, _PostImportFinder) for x in sys.meta_path) == 1


def test_late_arm_with_dependent_imported_fails_closed(monkeypatch):
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setitem(sys.modules, P.TARGET_MODULE, types.ModuleType("t"))
    monkeypatch.setitem(sys.modules, P.DEPENDENT_MODULE, types.ModuleType("d"))
    with pytest.raises(SystemExit, match="imported before"):
        P.install_post_import_hook()


# Synthetic target: the three anchors embedded in minimal runnable code, so
# the rewritten logic itself is exercised on CPU (the real file needs vllm).
_SYNTH = (
    "import logging\n"
    "def init_logger(n):\n"
    "    lg = logging.getLogger(n)\n"
    "    lg.info_once = lg.info\n"
    "    return lg\n"
    + P.EDITS[0][1]
    + "class SparseMLAPrefillMetadata: pass\n"
    "class _Cfg:\n"
    "    def __init__(self, hs):\n"
    "        self.attention_config = type('A', (), {'hisparse_config': hs})()\n"
    "class FlashInferMLASparseMetadataBuilder:\n"
    "    hisparse_supports_multi_token_decode = False\n"
    "    def __init__(self, hs):\n"
    "        self.vllm_config = _Cfg(hs)\n"
    "    def _init_reorder_batch_threshold(self, reorder_batch_threshold=128,\n"
    "                                      supports_spec_as_decode=True):\n"
    + P.EDITS[1][1]
    + "        return reorder_batch_threshold, supports_spec_as_decode\n"
    "    def build(self, num_decodes, num_decode_tokens, decode_max_query_len):\n"
    + P.EDITS[2][1]
    + "        return prefill_max_seq_len\n"
    "class OtherBuilder(FlashInferMLASparseMetadataBuilder):\n"
    "    pass\n"
)


def _fake_runtime(monkeypatch, tmp_path, src=None):
    path = tmp_path / "runtime.py"
    path.write_text(RUNTIME_FIXTURE.read_text() if src is None else src)
    rt = types.ModuleType(P.RUNTIME_MODULE)
    rt.__file__ = str(path)
    _defs(RUNTIME_FIXTURE, "_get_max_decode_query_len", "_get_max_swap_rows",
          ns=rt.__dict__)
    monkeypatch.setitem(sys.modules, P.RUNTIME_MODULE, rt)
    return rt


def _apply_synthetic(monkeypatch, tmp_path):
    rt = _fake_runtime(monkeypatch, tmp_path)
    path = tmp_path / "sparse_mla_attention.py"
    path.write_text(_SYNTH)
    mod = types.ModuleType(P.TARGET_MODULE)
    mod.__file__ = str(path)
    mod.__dict__["__name__"] = P.DEPENDENT_MODULE  # classes' __module__
    exec(compile(_SYNTH, str(path), "exec"), mod.__dict__)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: True)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.30.0"))
    assert P.apply(mod) is True
    assert rt._get_max_swap_rows(_cfg(3)) == 5
    return mod


def test_runtime_drift_fails_closed_before_any_rewrite(monkeypatch, tmp_path):
    rt = _fake_runtime(monkeypatch, tmp_path, src="drifted\n")
    path = tmp_path / "sparse_mla_attention.py"
    path.write_text(_SYNTH)
    mod = types.ModuleType(P.TARGET_MODULE)
    mod.__file__ = str(path)
    exec(compile(_SYNTH, str(path), "exec"), mod.__dict__)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: True)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.30.0"))
    with pytest.raises(P.PatchDriftError, match="runtime.py"):
        P.apply(mod)
    assert not hasattr(mod, P.MARKER_ATTR) and not hasattr(mod, "_SUFFIX_HISPARSE_MTP_STATS")
    assert rt._get_max_swap_rows(_cfg(3)) == 4


def test_apply_semantics_on_synthetic(monkeypatch, tmp_path):
    mod = _apply_synthetic(monkeypatch, tmp_path)
    assert getattr(mod, P.MARKER_ATTR) == P.PATCH_REVISION
    B = mod.FlashInferMLASparseMetadataBuilder
    # HiSparse on: SM120 builder keeps spec-as-decode (threshold 1 -> the
    # base class then raises it to 1+k); a subclass (SM90) does not.
    assert B(hs=object())._init_reorder_batch_threshold() == (1, True)
    assert mod.OtherBuilder(hs=object())._init_reorder_batch_threshold() == (1, False)
    # HiSparse off: untouched.
    assert B(hs=None)._init_reorder_batch_threshold() == (128, True)
    b = B(hs=object())
    b._init_reorder_batch_threshold()
    b.build(0, 0, 0)
    b.build(4, 4, 1)
    assert mod._SUFFIX_HISPARSE_MTP_STATS["multi_token_decode_builds"] == 0
    b.build(2, 8, 4)
    b.build(1, 4, 4)
    assert mod._SUFFIX_HISPARSE_MTP_STATS == {
        "multi_token_decode_builds": 2, "max_decode_query_len": 4}
    # Traceback-visible source: linecache-registered rewrite.
    import linecache
    fname = mod._suffix_hisparse_mtp_mark.__code__.co_filename
    assert P.PATCH_NAME in fname and P.HELPER_TAG in "".join(linecache.getlines(fname))
    assert P.apply(mod) is True  # idempotent (marker)


def test_apply_refuses_other_vllm(monkeypatch, tmp_path):
    path = tmp_path / "x.py"
    path.write_text(_SYNTH)
    mod = types.ModuleType("x")
    mod.__file__ = str(path)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: True)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.31.0"))
    with pytest.raises(P.PatchDriftError, match="0.31.0"):
        P.apply(mod)
    with pytest.raises(SystemExit, match="FAILED"):
        P._hook_callback(mod)


def _run(outputs, **kw):
    r = {"patched": None, "impls": ["FlashInferMLASparseSM120Impl"],
         "mtp_decode_builds": 0, "spills": 3, "outputs": outputs}
    r.update(kw)
    return r


def test_oracle_verdict():
    good = {"target_1": [1, 2], "pressure_0": [3], "target_2": [1, 2]}
    other = {"target_1": [1, 9], "pressure_0": [3], "target_2": [1, 9]}
    ref, stock = _run(good, spills=None), _run(good)
    patched = _run(good, patched="r", mtp_decode_builds=5)
    assert oracle.verdict(ref, stock, patched) == (0, [])
    # Gate is patched == ref; a diverging (completed) stock is only noted.
    rc, why = oracle.verdict(ref, _run(other), patched)
    assert rc == 0 and why == ["stock != ref at ['target_1@1', 'target_2@1']"]
    assert oracle.verdict(ref, stock, dict(patched, mtp_decode_builds=0))[0] == 1
    assert oracle.verdict(ref, stock, dict(patched, spills=0))[0] == 1
    assert oracle.verdict(_run(other), stock, patched)[0] == 2
    # Stock crash (the silicon case) is reported, never blocks judging patched.
    crashed = {"mode": "stock", "error": "exit 1: ValueError: stop=5, capacity=4."}
    rc, why = oracle.verdict(ref, crashed, patched)
    assert rc == 0 and "stock crashed" in why[0]
    rc, why = oracle.verdict(_run(other), crashed, patched)
    assert rc == 1 and "patched != ref" in why[0] and "stock crashed" in why[-1]
    assert oracle.verdict(ref, stock, {"mode": "patched", "error": "x"})[0] == 1
    assert oracle.verdict({"mode": "ref", "error": "x"}, stock, patched)[0] == 2
    bad = dict(patched, outputs=dict(good, target_2=[1, 3]))
    assert oracle.verdict(ref, stock, bad)[0] == 1


def test_oracle_main_runs_patched_after_stock_crash_for_each_k(monkeypatch, capsys):
    good = {"target_1": [1, 2], "target_2": [1, 2]}
    calls = []

    def fake(mode, argv):
        calls.append((mode, argv[-2:]))
        if mode == "stock":
            return {"mode": mode, "error": "exit 1: ValueError: capacity"}
        return _run(good, spills=3 if mode == "patched" else None,
                    patched="r" if mode == "patched" else None,
                    mtp_decode_builds=4 if mode == "patched" else 0,
                    max_decode_query_len=4)

    monkeypatch.setattr(oracle, "_run_child", fake)
    assert oracle.main(["--k", "3", "5", "--gpu-blocks", "200"]) == 0
    assert calls == [(m, ["--k", k]) for k in ("3", "5")
                     for m in ("ref", "stock", "patched")]
    out = capsys.readouterr().out
    assert "k=3 stock: CRASHED" in out and "k=5 PASS: stock crashed" in out
    # argparse: the appended --k overrides the sweep list in the child.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+")
    assert ap.parse_args(["--k", "3", "5", "--k", "5"]).k == [5]
