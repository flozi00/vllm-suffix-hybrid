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
    monkeypatch.delenv(P.PREFETCH_ENV, raising=False)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))


def test_fixture_is_pinned_wheel_copy():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert hashlib.sha256(RUNTIME_FIXTURE.read_bytes()).hexdigest() == RUNTIME_SHA256


def _defs(path, *names, ns=None):
    """exec the named top-level functions / class methods of a fixture."""
    import __future__
    import ast
    tree = ast.parse(path.read_text())
    ns = {} if ns is None else ns
    ns.setdefault("VllmConfig", object)  # annotation (runtime.py has no pep563 here)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            # Postponed annotations: CI (Python 3.11) evaluates them eagerly,
            # and fixtures annotate with names (torch, Any) not in ns.
            exec(compile(ast.Module([node], []), str(path), "exec",
                         flags=__future__.annotations.compiler_flag), ns)
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
    assert applied == ["helper_block", "hisparse_spec_as_decode", "build_marker",
                       "hisparse_prefill_without_backend", "prepare_metadata_guard"]
    for _name, old, rep, _c in P.EDITS:
        assert new.count(rep) == 1
    assert "if not self.hisparse_supports_multi_token_decode:" not in new
    assert new.count(P.HELPER_TAG) == 3
    # Byte-exact inverse: nothing outside the hunks changed.
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
        P.patch_source(src.replace(old, old[:-1] + " \n"))
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
    "        self._prefill_backend = None\n"
    "    def _init_reorder_batch_threshold(self, reorder_batch_threshold=128,\n"
    "                                      supports_spec_as_decode=True):\n"
    + P.EDITS[1][1]
    + "        return reorder_batch_threshold, supports_spec_as_decode\n"
    "    def build(self, num_decodes, num_decode_tokens, decode_max_query_len):\n"
    + P.EDITS[2][1]
    + "        num_prefills = 0\n"
    + P.EDITS[3][1]
    + "            prefill = 1\n"
    + P.EDITS[4][1]
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


UTILS_FIXTURE = FIXTURES / "attention_backends_utils.py"  # v1/attention/backends/utils.py
UTILS_SHA256 = "fbc221360d2d54cb21ef3f625dc8711500de345a355801548b4337bb9b8abee0"


def _build_fn(src):
    """The real SparseMLACommonMetadataBuilder.build (+ helpers) from src."""
    import torch
    ns = {"torch": torch, "T": object, "_suffix_hisparse_mtp_mark": None,
          "SparseMLAPrefillMetadata": types.SimpleNamespace,
          "build_hisparse_prefill_staging_plan":
              lambda bt, sl, bs, cap: types.SimpleNamespace(block_table=bt, cap=cap)}
    _defs(UTILS_FIXTURE, "split_decodes_and_prefills", ns=ns)
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "build", "_build_prefill_fields", "_use_dense_mha_prefill"):
            node.decorator_list = []
            exec(compile(ast.Module([node], []), "sparse_mla", "exec"), ns)
    return ns


def test_silicon_staging_plan_assert_replays_on_cpu_and_patch_fixes_it():
    """target_2 of the oracle: 4096-token prompt again, 63 blocks prefix-hit
    (pages spilled to host), 64 new tokens -> one prefill, not all resident.
    SM120 has no MLA prefill backend -> stock build() leaves prefill=None ->
    index_group.stage_prefill_rows asserts (the silicon AssertionError)."""
    import hashlib as h
    import torch
    assert h.sha256(UTILS_FIXTURE.read_bytes()).hexdigest() == UTILS_SHA256
    sm120 = (FIXTURES / "flashinfer_mla_sparse_sm120.py").read_text()
    assert sm120.count("    supports_dense_mha_prefill = False\n") == 1
    assert sm120.count("if num_decode_tokens == 0 and cache.all_context_pages_resident:") == 1
    src = FIXTURE.read_text()
    assert src.count("layer_prefill_backend.clone() if layer_prefill_backend "
                     "is not None else None") == 1

    def make_self(ns):
        s = types.SimpleNamespace(
            _build_prefill_fields=ns["_build_prefill_fields"],
            _prefill_backend=None,  # SM120 (mla_attention.py:600-607)
            reorder_batch_threshold=4, use_pcp=False, require_uniform_decodes=False,
            kv_cache_spec=types.SimpleNamespace(block_size=64),
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
            topk_tokens=2048, topk_mask_workspace=None, dcp_world_size=1,
            cp_kv_cache_interleave_size=1, metadata_cls=types.SimpleNamespace,
            vllm_config=types.SimpleNamespace(attention_config=types.SimpleNamespace(
                hisparse_config=object(), sparse_mla_force_mqa=False)),
            _build_req_id_per_token=lambda cm: torch.zeros(64, dtype=torch.int32),
            _build_chunked_context_fields=lambda *a: None)
        return s

    qsl = torch.tensor([0, 64], dtype=torch.int32)
    cm = types.SimpleNamespace(
        num_reqs=1, num_actual_tokens=64, max_query_len=64, max_seq_len=4096,
        query_start_loc=qsl, query_start_loc_cpu=qsl, max_logits_per_req=None,
        seq_lens=torch.tensor([4096]), seq_lens_cpu_upper_bound=torch.tensor([4096]),
        block_table_tensor=torch.zeros(1, 66, dtype=torch.int32),
        slot_mapping=None, is_prefilling=torch.tensor([True]))

    ig = _defs(FIXTURES / "index_group.py", "stage_prefill_rows")
    cache = types.SimpleNamespace(view=None, block_table=None,
                                  runtime=types.SimpleNamespace(
                                      gather_prefill_cache=lambda kv, plan, **k: "staged"))
    group = types.SimpleNamespace(cache=lambda i: cache)

    def forward_stage(md):  # flashinfer_mla_sparse_sm120.py:142-165
        assert md.num_decode_tokens < md.num_actual_tokens
        return ig["stage_prefill_rows"](group, 0, None, md)

    stock = _build_fn(src)
    md = stock["build"](make_self(stock), 0, cm)
    assert (md.num_decodes, md.num_prefills, md.prefill) == (0, 1, None)
    with pytest.raises(AssertionError):
        forward_stage(md)

    patched = _build_fn(P.patch_source(src)[0])
    md = patched["build"](make_self(patched), 0, cm)
    assert md.prefill.host_staging_plan.cap == 64  # ceil(4096/64) blocks
    staged, bt, _ = forward_stage(md)
    assert staged == "staged" and bt is md.prefill.host_staging_plan.block_table
    # With a prefill backend the section is unchanged and still prepared.
    seen = []
    s = make_self(patched)
    s._prefill_backend = types.SimpleNamespace(prepare_metadata=seen.append)
    assert patched["build"](s, 0, cm).prefill is seen[0]


def test_oracle_error_names_the_prompt_being_generated(monkeypatch):
    import subprocess as sp

    class Proc:
        pid, returncode = 1, 1
        stdout = iter([f"{oracle.MARK} patched: target_1 start\n",
                       f"{oracle.MARK} patched: target_1 done (32 tokens, 0.8s)\n",
                       f"{oracle.MARK} patched: target_2 start\n"])
        stderr = iter(["[rank0]: AssertionError\n"])

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(sp, "Popen", lambda *a, **k: Proc())
    monkeypatch.setattr(oracle.os, "killpg", lambda *a: None)
    r = oracle._run_child("patched", [])
    assert r["error"] == "exit 1 during target_2: [rank0]: AssertionError"


def _pool_ns(codes, rank=1, world=2):
    """Patched allocate_hisparse_host_pools against fakes: TP group, region,
    cudart returning `codes` in order (then 0)."""
    events, codes = [], list(codes)

    class Group:
        world_size, rank_in_group = world, rank

        def barrier(self):
            events.append("barrier")

    class Code:
        def __init__(self, v):
            self.value = v

    class Cudart:
        def cudaHostRegister(self, ptr, n, flags):
            events.append(("register", ptr, n, flags))
            return Code(codes.pop(0) if codes else 0)

        def cudaGetLastError(self):
            events.append("drain")

    class Buf:
        def __getitem__(self, s):
            return types.SimpleNamespace(data_ptr=lambda: 4096 + s.start,
                                         nbytes=s.stop - s.start)

    class Region:
        def __init__(self, **kw):
            self.base_tensor, self.pinned_addresses, self.is_pinned = Buf(), [], False

        def create_next_canonical_view(self, size):
            return types.SimpleNamespace(view=lambda *_: size)

        def cleanup(self):
            events.append("cleanup")

    import math
    import mmap
    ns = _defs(RUNTIME_FIXTURE, "_hisparse_registration_ranges",
               ns={"HOST_REGISTER_CHUNK_BYTES": 256 * 2**30})
    ns.update(math=math, mmap=mmap,
              SharedOffloadRegion=Region, get_tp_group=Group,
              check_hisparse_host_memory=None, allocate_pinned_host_pool=None,
              torch=types.SimpleNamespace(cuda=types.SimpleNamespace(cudart=Cudart)),
              time=types.SimpleNamespace(sleep=lambda s: events.append(("sleep", s))),
              logger=types.SimpleNamespace(warning=lambda *a: events.append("warn")))
    exec(P.patch_runtime_source(RUNTIME_FIXTURE.read_text()), ns)
    cfg = types.SimpleNamespace(instance_id="x", parallel_config=types.SimpleNamespace(
        data_parallel_index=0))
    # prod geometry scaled down: 352-B rows x 64 tokens, page-rounded stride
    call = lambda: ns["allocate_hisparse_host_pools"](  # noqa: E731
        cfg, [22528 * 8, 22528 * 8], 8, 45056, use_shared_host_pool=True)
    return call, events


def test_shared_pool_pins_in_rank_turns():
    call, events = _pool_ns([])
    pools, private, region = call()
    assert pools == [22528 * 8] * 2 and private == []
    # rank 1 of 2: registers the whole pool (one range) only in ITS turn.
    assert events == ["barrier", ("register", 4096, 8 * 45056, 0), "barrier"]
    assert region.pinned_addresses == [4096] and region.is_pinned


def test_shared_pool_retries_then_reports_the_numeric_code():
    call, events = _pool_ns([2, 0], rank=0)
    call()
    assert events.count("drain") == 1 and ("sleep", 2.0) in events
    call, events = _pool_ns([2, 2, 2], rank=0, world=3)
    with pytest.raises(RuntimeError, match=r"code=2 cudaErrorMemoryAllocation"):
        call()
    # every rank still reaches all world_size barriers (no peer hang), then
    # the region is cleaned up exactly once.
    assert events.count("barrier") == 3 and events[-1] == "cleanup"
    assert sum(isinstance(e, tuple) and e[0] == "register" for e in events) == 3


def test_pool_anchor_drift():
    src = RUNTIME_FIXTURE.read_text()
    with pytest.raises(P.PatchDriftError, match="pool_register_loop: expected 1, found 0"):
        P.patch_runtime_source(src.replace(P.POOL_OLD, P.POOL_NEW))
    with pytest.raises(P.PatchDriftError, match="pool_call_site: expected 1, found 0"):
        P.patch_runtime_source(src.replace(P.POOL_CALL_SITE, ""))


def test_oracle_tp2_requires_the_shared_host_pool():
    outs = {"target_1": [1], "target_2": [1]}
    ref = {"mode": "ref", "outputs": outs}
    ok = {"mode": "patched", "tp": 2, "outputs": outs, "patched": "r", "spills": 3,
          "impls": ["FlashInferMLASparseSM120Impl"], "mtp_decode_builds": 4,
          "shared_host_pool": [True, True]}
    stock = {"mode": "stock", "error": "exit 1: RuntimeError: cudaHostRegister failed"}
    assert oracle.verdict(ref, stock, ok)[0] == 0
    rc, why = oracle.verdict(ref, stock, dict(ok, shared_host_pool=[True, False]))
    assert rc == 1 and "shared HiSparse host pool" in why[0]
    assert oracle.verdict(ref, stock, dict(ok, tp=1, shared_host_pool=[False]))[0] == 0


# --- follower prefetch under MTP (SUFFIX_SM120_HISPARSE_PREFETCH) ----------

def _runtime_methods(src):
    import ast
    cls = next(n for n in ast.parse(src).body
               if isinstance(n, ast.ClassDef) and n.name == "HiSparseRuntime")
    import textwrap
    lines = src.splitlines(keepends=True)
    return {n.name: textwrap.dedent("".join(lines[n.lineno - 1:n.end_lineno]))
            for n in cls.body if isinstance(n, ast.FunctionDef)}


class _Replay:
    """HiSparseRuntime methods from runtime.py on fake streams/events. Logs
    every swap (runtime, rows) and checks at each wait that the awaited event
    was recorded after this runtime's rows for the current step were copied
    (copy stream is in-order, so record-after-swap == rows ready)."""

    def __init__(self, methods):
        import __future__
        import torch
        self.log = []
        self.stats = {"follower_prefetched": 0, "follower_staged": 0}
        replay = self

        class Event:
            def record(self, stream):
                self.covers = {e for e in replay.log if e[0] == "swap"}

        class Stream:
            def wait_event(self, ev):
                replay.waited = ev

            def wait_stream(self, s):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        compute = Stream()
        ns = {"torch": types.SimpleNamespace(Event=Event, Tensor=torch.Tensor),
              "current_stream": lambda: compute,
              "in_piecewise_cudagraph": lambda: False,
              P.PREFETCH_STATS: self.stats}
        body = {"_suffix_prefetched_through": 0}
        for name, src in methods.items():
            if name == "__init__":
                continue
            exec(compile(src, "rt", "exec", __future__.annotations.compiler_flag,
                         dont_inherit=True), ns, body)
        body["_resolve_residency"] = lambda rt, **kw: None
        body["_swap_rows"] = lambda rt, rows: replay.log.append(
            ("swap", rt.name, (rows.start, rows.stop)))
        RT = type("RT", (), body)
        group = types.SimpleNamespace(
            copy_stream=Stream(), logical_topk_ready=None, followers=[],
            shared_topk=types.SimpleNamespace(
                physical_topk_indices=torch.zeros(64, 1),
                valid_topk_counts=torch.zeros(64)))
        self.layers = []
        for i, name in enumerate(("leader", "f1", "f2")):
            rt = RT()
            rt.name, rt.is_group_leader, rt.index_group = name, i == 0, group
            rt._swap_step, rt._swap_staged, rt._layer_ready_event = 0, False, None
            rt.hot = types.SimpleNamespace(block_size=64, attention_block_stride=64)
            if i:
                group.followers.append(rt)
            else:
                group.leader = rt
            self.layers.append(rt)

    def forward(self, n, steps, *, decode_batch, num_actual, num_decode):
        """One forward: every layer (leader first) swaps `steps` x n rows,
        like index_group.convert_decode_logical_to_physical_topk."""
        import torch
        res = types.SimpleNamespace(decode_batch=decode_batch,
                                    num_actual_tokens=num_actual,
                                    num_decode_tokens=num_decode)
        for rt in self.layers:
            rt.begin_forward()
        for rt in self.layers:
            for step in range(steps):
                rt.swap_in(resident=res, req_id_per_token=None, block_table=None,
                           logical_topk_indices=torch.zeros(n, 1), block_size=64)
                rows = (step * n, step * n + n)
                assert self.waited is rt._layer_ready_event
                assert ("swap", rt.name, rows) in self.waited.covers
        return sorted(self.log)


def _replays():
    src = RUNTIME_FIXTURE.read_text()
    stock = _runtime_methods(src)
    patched = dict(stock, **P.patch_prefetch_source(src))
    return _Replay(stock), _Replay(patched)


def test_prefetch_pure_multi_token_decode_skips_follower_restage():
    """k=5, 4 decodes: 6 verify steps x 4 rows. Stock re-stages each follower
    every step (critical path); patched: the leader prefetches them all, same
    swaps (runtime, rows) exactly once each."""
    stock, patched = _replays()
    kw = dict(decode_batch=False, num_actual=24, num_decode=24)  # max_q_len 6
    s_log, p_log = stock.forward(4, 6, **kw), patched.forward(4, 6, **kw)
    assert s_log == p_log and len(p_log) == 3 * 6 == len(set(p_log))
    assert patched.stats == {"follower_prefetched": 12, "follower_staged": 0}
    # Next forward resets the mark (no stale coverage across batches).
    patched.log.clear()
    patched.forward(4, 6, decode_batch=False, num_actual=30, num_decode=24)
    assert patched.stats["follower_staged"] == 12


@pytest.mark.parametrize("kw,steps", [
    (dict(decode_batch=False, num_actual=30, num_decode=24), 6),  # mixed batch
    (dict(decode_batch=False, num_actual=5, num_decode=0), 1),    # pure prefill
    (dict(decode_batch=True, num_actual=4, num_decode=4), 1),     # q_len=1 (stock prefetch)
])
def test_prefetch_other_batches_unchanged(kw, steps):
    stock, patched = _replays()
    n = 4 if kw["num_decode"] else 5
    assert stock.forward(n, steps, **kw) == patched.forward(n, steps, **kw)
    staged = patched.stats["follower_staged"]
    assert staged == (0 if kw["decode_batch"] else 2 * steps)


def test_prefetch_anchor_drift():
    src = RUNTIME_FIXTURE.read_text()
    assert set(P.patch_prefetch_source(src)) == {e[0] for e in P.PREFETCH_EDITS}
    for name, old, _new in P.PREFETCH_EDITS:
        with pytest.raises(P.PatchDriftError, match=f"{name}: expected 1, found 0"):
            P.patch_prefetch_source(src.replace(old, old[:-1] + " \n"))
        with pytest.raises(P.PatchDriftError, match=f"{name}: expected 1, found 2"):
            P.patch_prefetch_source(src + "\n" + old)
    # Hunk moved out of its method (e.g. into HiSparseCacheHandle) = drift.
    old = P.PREFETCH_EDITS[2][1]
    moved = src.replace(old, "        pass\n").replace(
        "class HiSparseCacheHandle:\n", "class HiSparseCacheHandle:\n"
        "    def _x(self):\n" + old)
    with pytest.raises(P.PatchDriftError, match="not inside HiSparseRuntime.swap_in"):
        P.patch_prefetch_source(moved)


def _prefetch_runtime(monkeypatch, tmp_path):
    rt = _fake_runtime(monkeypatch, tmp_path)
    rt.HiSparseRuntime = type("HiSparseRuntime", (), {
        n: (lambda self: "stock") for n, *_ in P.PREFETCH_EDITS})
    return rt


def test_apply_installs_prefetch_only_with_its_gate(monkeypatch, tmp_path):
    mod = _apply_synthetic(monkeypatch, tmp_path)  # prefetch gate unset
    rt = sys.modules[P.RUNTIME_MODULE]
    assert not hasattr(rt, P.PREFETCH_STATS)
    rt = _prefetch_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv(P.PREFETCH_ENV, "1")
    delattr(mod, P.MARKER_ATTR)
    path = Path(mod.__file__)
    exec(compile(_SYNTH, str(path), "exec"), mod.__dict__)  # fresh stock source
    assert P.apply(mod) is True
    cls = rt.HiSparseRuntime
    assert rt.__dict__[P.PREFETCH_STATS] == {"follower_prefetched": 0,
                                               "follower_staged": 0}
    assert cls._suffix_prefetched_through == 0
    for name, *_ in P.PREFETCH_EDITS:
        fn = getattr(cls, name)
        assert fn.__globals__ is rt.__dict__ and P.PATCH_NAME in fn.__code__.co_filename
    obj = cls()
    obj._suffix_prefetched_through = 7
    obj.begin_forward()
    assert (obj._swap_step, obj._suffix_prefetched_through) == (0, 0)


def test_prefetch_drift_fails_closed_before_any_rewrite(monkeypatch, tmp_path):
    src = RUNTIME_FIXTURE.read_text().replace(P.PREFETCH_EDITS[1][1], "")
    rt = _fake_runtime(monkeypatch, tmp_path, src=src)
    rt.HiSparseRuntime = type("HiSparseRuntime", (), {})
    path = tmp_path / "sparse_mla_attention.py"
    path.write_text(_SYNTH)
    mod = types.ModuleType(P.TARGET_MODULE)
    mod.__file__ = str(path)
    exec(compile(_SYNTH, str(path), "exec"), mod.__dict__)
    monkeypatch.setenv(P.GATE_ENV, "1")
    monkeypatch.setenv(P.PREFETCH_ENV, "1")
    monkeypatch.setattr(P, "is_sm120", lambda cap=None: True)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.30.0"))
    with pytest.raises(P.PatchDriftError, match="_resolve_and_stage_group"):
        P.apply(mod)
    assert not hasattr(mod, P.MARKER_ATTR) and rt._get_max_swap_rows(_cfg(3)) == 4
    assert not hasattr(rt, P.PREFETCH_STATS)


# --- oracle --multi ---------------------------------------------------------

def test_multi_layout_matches_vllm_rules():
    # Silicon: 2 layers + MTP (every layer an indexer) -> 8 KV groups.
    for dt in ("fp8_ds_mla", "nvfp4_ds_mla"):
        g = oracle.hisparse_hot_groups(2, kv_dtype=dt)
        assert g == [[0], [1], [2]] and 2 + 2 * len(g) == 8
    # --multi: freq 4 / offset 3 -> indexers 0,1,2,6 (+MTP 8), followers
    # 3,4,5,7 (deepseek_v2.py: max(l - offset + 1, 0) % freq != 0 skips).
    assert oracle.index_leaders(8, 4, 3) == [True, True, True, False, False,
                                             False, True, False, True]
    g = oracle.hisparse_hot_groups(8, 4, 3, "nvfp4_ds_mla")
    assert g == [[0], [1], [2, 3, 4, 5], [6, 7], [8]] and 2 + 2 * len(g) == 12
    # Packing: a small source page lets whole units share a hot group.
    oracle.SOURCE_ROW["tiny"] = 8
    try:
        assert oracle.hisparse_hot_groups(3, kv_dtype="tiny") == [[0, 1, 2, 3]]
    finally:
        del oracle.SOURCE_ROW["tiny"]


def test_multi_gpu_blocks_forces_host_reads_but_admits():
    prompts = oracle.multi_prompts()
    mml = max(len(p) + n for _, p, n in prompts) + 64
    r = len(oracle.hisparse_hot_groups(8, 4, 3, "nvfp4_ds_mla"))
    blocks = oracle.multi_gpu_blocks(5, r, mml)
    pages = -(-mml // 64)
    hot = r * (5 + 2) * 2048 // 64
    one_request = (1 + r) * pages + hot           # indexer + resident + hot
    watermark = max(hot, blocks // 10)            # coordinator.py:158
    first_four = sum((1 + r) * -(-len(p) // 64) for _, p, _ in prompts[:4])
    assert one_request < blocks                   # admission progresses
    assert first_four > blocks - watermark        # -> reads from host


def test_multi_prompts_shape():
    ps = oracle.multi_prompts()
    assert len(ps) == 8 and len({n for n, *_ in ps}) == 8
    d = {n: (p, m) for n, p, m in ps}
    assert [len(d[n][0]) for n in ("target", "long_3000", "long_2500", "short_700")] \
        == [4096, 3000, 2500, 700]
    for name, tail in (("share_tail3", 3), ("share_tail40", 40)):
        assert d[name][0][:2048] == d["target"][0][:2048]
        assert len(d[name][0]) == 2048 + tail
    assert (len(d["prefill_25"][0]), d["prefill_25"][1]) == (25, 1)
    assert d["target_again"][0] == d["target"][0]
    assert len({m for _, _, m in ps}) == 8        # staggered finishes
    assert ps == oracle.multi_prompts()           # deterministic


def _mrun(mode, outputs, **kw):
    r = _run(outputs, mode=mode, patched="r" if mode != "ref" else None,
             mtp_decode_builds=0 if mode == "ref" else 9,
             spills=None if mode == "ref" else 4, max_decode_query_len=6,
             prefetch={"follower_prefetched": 40, "follower_staged": 0}
             if mode == "prefetch" else None)
    r.update(kw)
    return r


def test_oracle_verdict_multi():
    good = {"target": [1, 2, 3], "target_again": [1, 2], "x": [5]}
    ref, pat, pf = (_mrun(m, good) for m in ("ref", "patched", "prefetch"))
    assert oracle.verdict_multi(ref, pat, pf) == (0, [])
    bad = dict(good, x=[6])
    rc, why = oracle.verdict_multi(ref, pat, _mrun("prefetch", bad))
    assert rc == 1 and why == ["prefetch != ref at ['x@0']",
                               "prefetch != patched at ['x@0']"]
    assert oracle.verdict_multi(ref, pat, dict(pf, prefetch={
        "follower_prefetched": 0, "follower_staged": 7}))[1] == [
        "prefetch: leader never prefetched follower rows"]
    assert oracle.verdict_multi(ref, dict(pat, spills=0), pf)[0] == 1
    assert oracle.verdict_multi(ref, dict(pat, mtp_decode_builds=0), pf)[0] == 1
    assert oracle.verdict_multi(ref, _mrun("patched", dict(good, target_again=[1, 9])),
                                pf)[0] == 1
    rc, why = oracle.verdict_multi(ref, {"mode": "patched", "error": "exit 1 during "
                                         "step7(run=4,pf=1,wait=4): boom"}, pf)
    assert rc == 1 and why == ["patched crashed (exit 1 during step7(run=4,pf=1,wait=4): boom)"]
    assert oracle.verdict_multi({"mode": "ref", "error": "x"}, pat, pf)[0] == 2
    assert oracle.verdict_multi({"mode": "ref", "error": "x"}, pat, dict(pf, patched=None))[0] == 1


def test_oracle_main_multi_boot_gate_argv(monkeypatch, capsys):
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n}\n") + 3], ns)
    argv, env = ns["_BOOT_GATES"]["glm_stack_multi_oracle"]
    assert argv[:2] == ["-m", "hisparse_mtp_patch.oracle"]
    assert env == {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1"}
    good = {"target": [1, 2], "target_again": [1, 2]}
    calls = []

    def fake(mode, a):
        calls.append((mode, a))
        return _mrun(mode, good, steps={"steps": 90, "mixed": 30, "max_running": 4},
                     peak_reserved_gib=7.5)

    monkeypatch.setattr(oracle, "_run_child", fake)
    assert oracle.main(argv[2:]) == 0
    assert [m for m, _ in calls] == ["ref", "patched", "prefetch"]
    assert all(a[-2:] == ["--k", "5"] and "--multi" in a for _, a in calls)
    out = capsys.readouterr().out
    assert "k=5 PASS" in out and '"mixed": 30' in out and "peak_reserved_gib=7.5" in out
    assert oracle._child_timeout(argv) == 1200 and oracle._child_timeout([]) == 600


def test_oracle_child_env_per_mode(monkeypatch):
    import subprocess as sp
    seen = {}

    class Proc:
        pid, returncode = 1, 0
        stdout = iter([oracle.RESULT + '{"mode": "x"}\n'])
        stderr = iter([])

        def wait(self, timeout=None):
            return 0

    def popen(cmd, env, **kw):
        seen[cmd[4]] = {k: v for k, v in env.items() if "HISPARSE" in k}
        return Proc()

    monkeypatch.setattr(sp, "Popen", popen)
    monkeypatch.setattr(oracle.os, "killpg", lambda *a: None)
    monkeypatch.setenv("SUFFIX_SM120_HISPARSE_PREFETCH", "1")  # never leaks
    for m in ("ref", "stock", "patched", "prefetch"):
        oracle._run_child(m, ["--multi"])
    assert seen == {"ref": {}, "stock": {},
                    "patched": {"SUFFIX_SM120_HISPARSE_MTP": "1"},
                    "prefetch": {"SUFFIX_SM120_HISPARSE_MTP": "1",
                                 "SUFFIX_SM120_HISPARSE_PREFETCH": "1"}}
