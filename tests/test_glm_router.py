# SPDX-License-Identifier: Apache-2.0
"""Our GLM MoE router (suffix_hybrid/kernels/glm_router.py): CPU contract tests. The Triton
kernel body runs through tests/triton_shim.py (one program at a time, bounds-checked pointers,
poisoned masked lanes); the selection is checked against an emulation of vLLM's CUDA
single_group_topk block kernel (per-lane sorted keys, placeholder refill, cg::reduce
butterfly). Silicon: the load oracle (bit-identity vs ops.grouped_topk) + boot gate
glm_router_bench."""
import importlib
import math
import os
import pathlib
import struct
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import triton_shim  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sm120"))


@pytest.fixture(scope="module")
def gr():
    with triton_shim.installed():
        sys.modules.pop("suffix_hybrid.kernels.glm_router", None)
        mod = importlib.import_module("suffix_hybrid.kernels.glm_router")
        assert isinstance(mod._router_kernel, triton_shim.JITFunction)
        yield mod
    sys.modules.pop("suffix_hybrid.kernels.glm_router", None)


# --- vLLM's CUDA kernel, structurally (csrc/libtorch_stable/moe/grouped_topk_kernels.cu,
# single_group_topk_block_kernel, non-hierarchical warp path, + moeTopKFuncs.cuh) ---------
def _f32(v: float) -> float:
    return struct.unpack("f", struct.pack("f", v))[0]


def _twiddle(v: float) -> int:  # cub::Traits<float>::TwiddleIn as unsigned 32
    b = struct.unpack("I", struct.pack("f", v))[0]
    return b ^ 0xFFFFFFFF if b >> 31 else b | 0x80000000


def cuda_single_group_topk(logits, bias, topk, renorm, rsf, lanes=32):
    """Row by row: preprocess_score, per-lane descending lists of NumChunks keys, topk rounds
    of warp max with the lane-list shift + (InvalidScore, idx[N-1]) refill, write_outputs
    with the shfl_xor butterfly sum."""
    m, n = logits.shape
    chunks = -(-n // lanes)
    t = torch.tanh(0.5 * logits.float()) * 0.5 + 0.5  # fp32 like 0.5f*tanhf(0.5f*x)+0.5f
    ws, ids = [], []
    for r in range(m):
        unb, sel = [0.0] * n, [float("-inf")] * n
        for e in range(n):
            x, b = float(logits[r, e]), float(bias[e])
            if math.isfinite(x) and math.isfinite(b):
                unb[e] = float(t[r, e])
                s = _f32(unb[e] + b)
                sel[e] = 0.0 if s == 0.0 else s
        heads = []
        for lane in range(lanes):
            idx = [c * lanes + lane for c in range(chunks)]
            keys = [((_twiddle(sel[e] if e < n else float("-inf")) << 32) | (65535 - e))
                    for e in idx]
            heads.append([sorted(keys, reverse=True), idx[-1]])
        top = []
        packed = None
        for kk in range(topk):
            for h in heads:
                if kk > 0 and packed == h[0][0]:
                    h[0] = h[0][1:] + [(_twiddle(float("-inf")) << 32) | (65535 - h[1])]
            packed = max(h[0][0] for h in heads)
            top.append(packed)
        lane_e = [65535 - (k & 0xFFFF) for k in top]
        fin = [(k >> 32) != _twiddle(float("-inf")) for k in top]
        u = torch.tensor([unb[e] if f and 0 <= e < n else 0.0 for e, f in zip(lane_e, fin)]
                         + [0.0] * (lanes - topk), dtype=torch.float32)
        for mask in (16, 8, 4, 2, 1):
            u = u + u[torch.arange(lanes) ^ mask]
        s = u[0]
        row = []
        for j in range(topk):
            if not any(fin):
                row.append(1.0 / topk if renorm else 0.0)
            elif fin[j]:
                sc = torch.tensor(rsf, dtype=torch.float32)
                if renorm:
                    sc = sc / (s + torch.tensor(1e-20, dtype=torch.float32))
                row.append(float(torch.tensor(unb[lane_e[j]], dtype=torch.float32) * sc))
            else:
                row.append(0.0)
        ws.append(row)
        ids.append(lane_e)
    return torch.tensor(ws, dtype=torch.float32), torch.tensor(ids, dtype=torch.int32)


def _bits_equal(a, b) -> bool:
    return torch.equal(a.float().view(torch.int32), b.float().view(torch.int32))


@pytest.mark.parametrize("seed", [0, 1])
def test_flat_key_order_equals_the_cuda_warp_algorithm(gr, seed):
    """256 experts x top-8 (GLM): reference_topk == the CUDA kernel's structure on random,
    tied, NaN / inf, all-invalid, few-finite, +-0 and saturated rows, for every bias set."""
    for lg, b in gr.adversarial_logits(256, "cpu", seed=seed):
        rw, rid = gr.reference_topk(lg, b, 8, True, 2.5)
        cw, cid = cuda_single_group_topk(lg, b, 8, True, 2.5)
        assert torch.equal(rid, cid)
        assert _bits_equal(rw, cw)
        rw, rid = gr.reference_topk(lg, b, 8, False, 1.0)
        cw, cid = cuda_single_group_topk(lg, b, 8, False, 1.0)
        assert torch.equal(rid, cid) and _bits_equal(rw, cw)


def _run_selection(gr, lg, b, bn, topk=8, renorm=True, rsf=2.5):
    n = lg.shape[1]
    sem = torch.zeros(64, dtype=torch.int32)
    plan = gr.Plan(False, 32, bn, 128, 1, 1)
    _, tw, ids = gr.run(None, torch.empty((n, 128), dtype=torch.bfloat16), b, plan, sem,
                        fuse=True, topk=topk, renorm=renorm, rsf=rsf, logits_in=lg.clone())
    assert int(sem.abs().sum()) == 0, "counters must be re-armed"
    return tw, ids


@pytest.mark.parametrize("bn", [16, 32, 64])
def test_kernel_selection_is_the_reference_bit_for_bit(gr, bn):
    for lg, b in gr.adversarial_logits(256, "cpu", seed=3):
        tw, ids = _run_selection(gr, lg, b, bn)
        rw, rid = gr.reference_topk(lg, b, 8, True, 2.5)
        assert torch.equal(ids, rid)
        assert _bits_equal(tw, rw)


def test_kernel_selection_without_renormalize(gr):
    lg, b = gr.adversarial_logits(256, "cpu", seed=4)[0]
    tw, ids = _run_selection(gr, lg, b, 32, renorm=False, rsf=1.5)
    rw, rid = gr.reference_topk(lg, b, 8, False, 1.5)
    assert torch.equal(ids, rid) and _bits_equal(tw, rw)


@pytest.mark.parametrize("m,mma,bm", [(1, False, 1), (3, False, 4), (6, False, 8),
                                      (12, True, 16), (24, True, 32), (40, True, 32),
                                      (40, True, 64)])
@pytest.mark.parametrize("n_split", [1, 4])
def test_gemv_and_fused_selection(gr, m, mma, bm, n_split):
    """Logits rel <= TOL vs fp64 (GEMV / tl.dot, split-K, M tiles), fused selection == the
    reference on those logits, counters re-armed, no write past M."""
    n, k, bn, bk = 64, 512, 16, 64
    g = torch.Generator().manual_seed(m * 10 + n_split)
    w = (torch.randn(n, k, generator=g, dtype=torch.float32) * 0.05).bfloat16()
    x = torch.randn(m, k, generator=g, dtype=torch.float32).bfloat16()
    bias = torch.randn(n, generator=g, dtype=torch.float32) * 1e-2
    kbs = k // bk // n_split
    plan = gr.Plan(mma, bm, bn, bk, kbs, n_split)
    sem = torch.zeros(64, dtype=torch.int32)
    lg, tw, ids = gr.run(x, w, bias, plan, sem, fuse=True, topk=8, renorm=True, rsf=2.5)
    assert gr.gemv_rel(lg, x, w) <= gr.TOL
    rw, rid = gr.reference_topk(lg, bias, 8, True, 2.5)
    assert torch.equal(ids, rid) and _bits_equal(tw, rw)
    assert int(sem.abs().sum()) == 0
    lg2, tw2, _ = gr.run(x, w, bias, plan, sem, fuse=False)
    assert tw2 is None and _bits_equal(lg2, lg)  # deterministic per plan


def test_plans_cover_every_bucket_with_nonempty_splits(gr):
    for m in gr.BUCKETS:
        ps = gr.plans(256, 6144, m)
        assert ps, m
        for p in ps:
            assert p.n_split * p.kb_split * p.bk >= 6144 > (p.n_split - 1) * p.kb_split * p.bk
            assert p.bm >= (m if not p.mma else min(m, p.bm)) and 256 % p.bn == 0
            assert (p.mma and p.bm >= 16) or (not p.mma and m <= 8 and p.bm >= m)


def test_decide(gr):
    p = gr.Plan(False, 8, 32, 256, 3, 8)
    inf = float("inf")
    rows = {6: {"stock": 11.0, "gemv": 5.0, "fused": 3.0}, 96: {"stock": 9.0, "gemv": 10.0,
                                                               "fused": 12.0}}
    r = gr.decide([6, 96], {6: p, 96: p}, rows, "auto", True)
    assert r.lookup(1) == ("fused", p) and r.lookup(96) == ("stock", None)
    assert r.lookup(97) == ("stock", None)
    assert gr.decide([6], {6: p}, rows, "gemv", True).lookup(6) == ("gemv", p)
    assert gr.decide([6], {6: p}, rows, "auto", False).lookup(6) == ("gemv", p)
    assert gr.decide([6], {6: p}, rows, "1", True).lookup(6) == ("fused", p)
    with pytest.raises(RuntimeError, match="no working fused plan"):
        gr.decide([6], {6: None}, {6: {"stock": 1.0, "gemv": inf, "fused": inf}}, "1", True)


def _pair(gr, **over):
    gate = SimpleNamespace(weight=SimpleNamespace(dtype=torch.bfloat16, shape=(256, 6144),
                                                  dim=lambda: 2, is_contiguous=lambda: True,
                                                  device=SimpleNamespace(type="cuda")),
                           bias=None, is_unquantized=True, out_dtype=torch.float32)
    router = type("GroupedTopKRouter", (), {})()
    router.__dict__.update(num_expert_group=1, topk_group=1, scoring_func="sigmoid",
                           e_score_correction_bias=torch.zeros(256, dtype=torch.float32),
                           top_k=8, eplb_state=None, num_fused_shared_experts=0)
    for k, v in over.items():
        (gate if hasattr(gate, k) else router).__dict__[k] = v
    return gate, router


def test_eligibility(gr):
    assert gr.eligible(*_pair(gr)) is None
    assert "fp32" in gr.eligible(*_pair(gr, out_dtype=None))
    assert "grouping" in gr.eligible(*_pair(gr, num_expert_group=8))
    assert "sigmoid" in gr.eligible(*_pair(gr, scoring_func="softmax"))
    assert "top_k" in gr.eligible(*_pair(gr, top_k=6))
    assert "EPLB" in gr.eligible(*_pair(gr, eplb_state=object()))
    assert "torch path" in gr.eligible(*_pair(gr), use_fused_grouped_topk=False)
    gate, router = _pair(gr)
    assert "GroupedTopKRouter" in gr.eligible(gate, SimpleNamespace(**router.__dict__))


def test_grouped_topk_wrapper_serves_only_its_own_logits(gr):
    calls = []

    def orig(**kw):
        calls.append(kw)
        return "stock"
    f = gr._grouped_topk(orig)
    lg, b = torch.zeros(2, 4), torch.zeros(4)
    sig = (8, True, 1, 1, "sigmoid", 2.5)
    kw = dict(hidden_states=None, topk=8, renormalize=True, num_expert_group=1, topk_group=1,
              scoring_func="sigmoid", routed_scaling_factor=2.5, e_score_correction_bias=b)
    gr._STASH[0] = (lg, "tw", "ids", sig, b.data_ptr())
    assert f(gating_output=lg, **kw) == ("tw", "ids") and not calls
    assert gr._STASH[0] is None
    gr._STASH[0] = (lg, "tw", "ids", sig, b.data_ptr())
    assert f(gating_output=lg.clone(), **kw) == "stock"         # other logits: stock
    gr._STASH[0] = (lg, "tw", "ids", sig, b.data_ptr())
    assert f(gating_output=lg, **{**kw, "topk": 6}) == "stock"   # other params: stock
    assert f(gating_output=lg, **kw) == "stock" and len(calls) == 3  # stash consumed


def test_forward_patch_falls_back_and_refuses_when_never_set_up(gr, monkeypatch):
    class Gate:
        def forward(self, x):
            return ("stock", None)
    monkeypatch.setitem(gr._STATE, "setup_ran", False)
    gr._patch_gate(SimpleNamespace(GateLinear=Gate))
    with pytest.raises(SystemExit, match="armed but no model"):
        Gate().forward(torch.zeros(1, 8))
    monkeypatch.setitem(gr._STATE, "setup_ran", True)
    assert Gate().forward(torch.zeros(1, 8)) == ("stock", None)  # ineligible gate: stock


def test_gate_off_installs_nothing_and_on_arms_one_finder(gr, monkeypatch):
    before = list(sys.meta_path)
    monkeypatch.delenv(gr.GATE, raising=False)
    gr.install_post_import_hook()
    assert sys.meta_path == before
    monkeypatch.setenv(gr.GATE, "bogus")
    with pytest.raises(SystemExit, match="want one of"):
        gr.install_post_import_hook()
    monkeypatch.setenv(gr.GATE, "auto")
    for name in gr.PATCHES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    gr.install_post_import_hook()
    try:
        assert len(sys.meta_path) == len(before) + 1
    finally:
        sys.meta_path[:] = before


def test_load_time_helpers_ignore_vllms_bf16_default_dtype(gr):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        sets = gr.adversarial_logits(256, "cpu")
        assert all(lg.dtype == torch.float32 and b.dtype == torch.float32 for lg, b in sets)
        rw, rid = gr.reference_topk(*sets[0], 8, True, 2.5)
        tw, ids = _run_selection(gr, *sets[0], 32)
    finally:
        torch.set_default_dtype(prev)
    assert rw.dtype == torch.float32 and torch.equal(ids, rid) and _bits_equal(tw, rw)


def test_boot_gate_is_allowlisted():
    src = (ROOT / "sitecustomize.py").read_text()
    assert '"glm_router_bench": (["-m", "suffix_hybrid.kernels.glm_router", "bench"], {})' in src
    assert 'SUFFIX_ROUTER_OURS' in src
