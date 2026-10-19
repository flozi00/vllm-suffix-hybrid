# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph capture discipline of the DSA indexer lane (suffix_hybrid/kernels/dsa_capture.py)
and its use by the Helion serving entries: no JIT inside a capture (an unseen
specialisation class is refused, a warmed one launches), warm-ups cover every
class the serving shapes can land in, and every autotune candidate / shipped
config stays inside the bounded space."""
import os

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")
torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsa_indexer as hdi  # noqa: E402
from suffix_hybrid.kernels import dsa_capture as cap  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
import test_helion_dsa_indexer as th  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh():
    cap.reset()
    yield
    cap.reset()


def test_int_classes_follow_tritons_specialisation():
    assert [cap._cls(v) for v in (0, 1, 2, 15, 16, 17, 32, 6, 48, 96, 192)] == [2, 1, 0, 0, 16, 0, 16, 0, 16, 16, 16]


def test_spec_key_sees_what_triton_specialises_on():
    a = torch.zeros((6, 64), dtype=torch.float32)
    base = cap.spec_key(a, 64, 7)
    assert cap.spec_key(torch.zeros((6, 64)), 64, 7) == base                 # same classes
    assert cap.spec_key(torch.zeros((6, 64)), 80, 7) == base                 # 64 -> 80: both % 16
    assert cap.spec_key(torch.zeros((6, 64)), 64, 9) == base                 # 7 -> 9: both "other"
    assert cap.spec_key(torch.zeros((6, 64)), 64, 16) != base                # "other" -> % 16
    assert cap.spec_key(torch.zeros((6, 64)), 1, 7) != base                  # == 1 is its own binary
    assert cap.spec_key(torch.zeros((16, 64)), 64, 7) != base                # a size crossing into % 16
    assert cap.spec_key(torch.zeros((6, 63)), 64, 7) != base                 # a stride crossing out of % 16
    assert cap.spec_key(torch.zeros((6, 64), dtype=torch.float16), 64, 7) != base
    assert cap.spec_key(True) != cap.spec_key(False)
    # 16-byte pointer alignment
    buf = torch.zeros(64, dtype=torch.float32)
    aligned = buf[(-buf.data_ptr() // 4) % 4:][:16]
    skew = buf[((-buf.data_ptr() // 4) % 4) + 1:][:16]
    assert aligned.data_ptr() % 16 == 0 and skew.data_ptr() % 16 != 0
    assert cap.spec_key(aligned) != cap.spec_key(skew)
    # the int32 / int64 index class (a view that spans more than 2^31 elements)
    huge = torch.zeros(8).as_strided((2 ** 31 + 5,), (0,))
    assert cap.spec_key(huge)[0][-1] is True and cap.spec_key(a)[0][-1] is False


def test_launch_refuses_an_unwarmed_class_inside_a_capture_only(monkeypatch):
    calls = []
    kern = lambda *a: calls.append(a) or "ran"  # noqa: E731
    x3, x16 = torch.zeros((3, 8)), torch.zeros((16, 8))
    monkeypatch.setattr(cap, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside a CUDA-graph capture"):
        cap.launch(kern, x3, 3)
    assert not calls and not cap.seen(kern, x3, 3)
    monkeypatch.setattr(cap, "capturing", lambda: False)
    assert cap.launch(kern, x3, 3) == "ran" and cap.seen(kern, x3, 3)
    monkeypatch.setattr(cap, "capturing", lambda: True)
    assert cap.launch(kern, torch.zeros((5, 8)), 5) == "ran"      # 3 -> 5: the same classes, warm
    with pytest.raises(RuntimeError, match="warm every shape class"):
        cap.launch(kern, x16, 16)                                   # a new class: refused, not compiled
    assert len(calls) == 2
    other = lambda *a: "other"  # noqa: E731
    with pytest.raises(RuntimeError):                                # a different kernel object is its own binary
        cap.launch(other, x3, 3)


def test_search_filter_clamps_every_candidate_into_the_bounded_space():
    helion = pytest.importorskip("helion")
    wild = helion.Config(block_sizes=[64], num_warps=32, num_stages=8, pid_type="persistent_interleaved",
                         range_warp_specializes=[True, None, False], min_tiles=4, split_div=2)
    got = cap.search_filter(wild).config
    assert got["pid_type"] == "flat" and got["num_warps"] == 8 and got["num_stages"] == 4
    assert got["range_warp_specializes"] == [False, None, False]
    assert got["block_sizes"] == [64] and got["min_tiles"] == 4 and got["split_div"] == 2   # tunables untouched
    ok = helion.Config(block_sizes=[32], num_warps=2, num_stages=3, pid_type="flat")
    assert cap.search_filter(ok).config == ok.config
    # the settings dict is what helion.kernel accepts
    helion.kernel(hdi.dsa_group_logits, autotune_effort="none", **cap.SEARCH_SETTINGS)


def test_serving_refuses_configs_outside_the_bounded_space():
    assert cap.serving_violation(None) is None and cap.serving_violation({}) is None
    assert cap.serving_violation({"pid_type": "flat", "num_warps": 4, "num_stages": 5}) is None   # shipped GLM configs
    assert "deadlock" in cap.serving_violation({"pid_type": "persistent_blocked"})
    assert "warp-specialised" in cap.serving_violation({"range_warp_specializes": [None, True]})


def _arm(monkeypatch, kernels):
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"block_sizes": [32]} if kernel in kernels else None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", th.make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)


@pytest.mark.parametrize("lens,nn", [([150], 5), ([200, 70], 6), ([120, 80, 60, 40, 30, 20, 10, 9], 6)])
def test_warmed_group_serves_every_batch_size_inside_a_capture(monkeypatch, lens, nn):
    """After the warm-up (as the first eager call does), group_logits for 1 .. 8
    request groups launches inside a (simulated) capture: no class is left to
    compile mid-capture, and the result is the spec."""
    _arm(monkeypatch, {hdi.GROUP})
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = th.decode_problem(32, lens, nn, 300, seed=6)
    cache = th.cache_of(kv, pg, prow).unsqueeze(2)
    hdi._warm(hdi.GROUP, 32, torch.device("cpu"), cache, 300, "auto", bt.shape[1], g=6)
    ref = hdi.decode_reference(q, w, kv, ks, bt, cl, out, pg, prow, sp, 2)
    monkeypatch.setattr(cap, "capturing", lambda: True)
    hdi.group_logits(q, w, cache, bt, cl, out, 300, 6)
    ok, msg = hdi.check(out, ref)
    assert ok, msg


def test_a_new_block_table_width_class_cannot_be_warmed_inside_a_capture(monkeypatch):
    _arm(monkeypatch, {hdi.GROUP})
    q, w, kv, ks, bt, cl, out, pg, prow, sp, _ = th.decode_problem(32, [200, 70], 6, 300, seed=6)
    cache = th.cache_of(kv, pg, prow).unsqueeze(2)
    hdi.group_logits(q, w, cache, bt, cl, out, 300, 6)           # eager: warms width class of bt.shape[1]
    wide = torch.zeros((bt.shape[0], 48), dtype=torch.int32)
    wide[:, :bt.shape[1]] = bt
    assert cap._cls(wide.shape[1]) != cap._cls(bt.shape[1])
    monkeypatch.setattr(cap, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside CUDA-graph capture"):
        hdi.group_logits(q, w, cache, wide, cl, out, 300, 6)


def test_shipped_configs_outside_the_bounded_space_refuse_to_serve(monkeypatch):
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"block_sizes": [32], "pid_type": "persistent_blocked"} if kernel == hdi.GROUP else None)
    monkeypatch.setattr(hdi, "_WARM", set())
    cache = torch.zeros((4, 64, 1, 132), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="pid_type"):
        hdi._warm(hdi.GROUP, 32, torch.device("cpu"), cache, 300, "auto", 8, g=6)


def test_group_bodies_have_a_bounded_config_space():
    """The row loop is rolled, not unrolled: one eviction / indexing entry per memory
    op of the per-row body, not G times as many (LFBO searches lists of ~8, not ~45)."""
    helion = pytest.importorskip("helion")
    import json
    from helion.runtime.ref_mode import RefMode

    def lens(body, args):
        spec = helion.kernel(body, ref_mode=RefMode.OFF, static_shapes=True).bind(args).config_spec
        c = json.loads(spec.default_config().to_json())
        return len(c["load_eviction_policies"]), len(c["indexing"])
    row = lens(hdi.dsa_decode_logits, th.decode_problem(32, [200, 70], 6, 300))
    grp = lens(hdi.dsa_group_logits, th.group_problem(32, [200, 70], 6, 300)[0])
    sel = lens(hdi.dsa_group_select, th.select_problem(32, [200, 70], 6, 300)[0])
    assert max(grp) <= 12 and max(sel) <= 14, (row, grp, sel)
