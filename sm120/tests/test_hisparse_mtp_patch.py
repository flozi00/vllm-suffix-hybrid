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


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(P.GATE_ENV, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


def test_fixture_is_pinned_wheel_copy():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


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


def _apply_synthetic(monkeypatch, tmp_path):
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
    return mod


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
    rc, why = oracle.verdict(ref, _run(other), patched)
    assert rc == 1 and "target_1@1" in why[0]
    assert oracle.verdict(ref, stock, dict(patched, mtp_decode_builds=0))[0] == 1
    assert oracle.verdict(ref, stock, dict(patched, spills=0))[0] == 1
    assert oracle.verdict(_run(other), stock, patched)[0] == 2
