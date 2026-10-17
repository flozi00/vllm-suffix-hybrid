# SPDX-License-Identifier: Apache-2.0
"""suffix_hybrid/helion_kernels/ds_mla.py: lab ABI (cases, replaces vs the
oxide PTX entries), the serving views (padded HiSparse block strides), the
SUFFIX_HELION_DS_MLA knob (oxide / helion / auto), the patch's dispatch,
the load-time oracle, and — with Helion installed — the body in Helion's
eager reference mode (HELION_INTERPRET=1) on tiny CPU tensors against the
oracle's reference (-1 tails/holes, an all -1 token, NaN SF bytes behind
the -1 clamp target, |q| > f16 max, split counts 1 / 3 / 40 of 64)."""
import os

os.environ.setdefault("HELION_INTERPRET", "1")  # before any helion import

import re  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sm120"))
import nvfp4_ds_mla_patch as P  # noqa: E402
from nvfp4_ds_mla_patch import oracle as O  # noqa: E402

from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import ds_mla as D  # noqa: E402

GLM = {"name": "glm", "architecture": "GlmMoeDsaForCausalLM", "heads": 8, "tp": 8}


def problem(seed=0, t=4, h=4, cap=40, bs=16, nb=6, pad=32):
    """(q, padded-stride cache, topk): row 1 all -1, row 2 a -1 tail, row 3
    holes; slot 0 (the kernel's clamp target for -1) holds NaN SF bytes and
    is never referenced; token 0 has |q| > f16 max."""
    g = torch.Generator().manual_seed(seed)
    n = nb * bs
    chan = torch.exp(torch.randn(D.DIM, generator=g) * 0.75)
    kv_c = (torch.randn(n, D.DIM, generator=g) * chan).bfloat16()
    kv_c[1, :16] = 5000.0  # saturating SF block (448)
    k_pe = (torch.randn(n, D.PE, generator=g) * 2.0).bfloat16()
    rows = O.quant_rows_ref(kv_c, k_pe)
    rows[0, 320:] = 0x7F  # e4m3 NaN
    stride = bs * D.ROW + pad
    raw = torch.zeros(nb * stride, dtype=torch.uint8)
    cache = raw.as_strided((nb, bs, D.ROW), (stride, D.ROW, 1))
    cache.copy_(rows.view(nb, bs, D.ROW))
    topk = torch.randint(1, n, (t, cap), generator=g, dtype=torch.int32)
    topk[1] = -1
    topk[2, 25:] = -1
    topk[3, ::3] = -1
    topk[3] = topk[3, torch.randperm(cap, generator=g)]
    q = torch.randn(t, h, D.DIM + D.PE, generator=g)
    q[0] *= 2.0 ** 15
    return q.bfloat16(), cache, topk


def args_of(q, cache, topk):
    out = torch.empty(q.shape[0], q.shape[1], D.DIM, dtype=torch.bfloat16)
    return (*D.views(q, cache, topk, out), O.SCALE)


# ---- pure python -----------------------------------------------------------
def test_cases_and_lab_abi():
    cs = D.cases(GLM)
    assert [c["numtokens"] for c in cs] == list(D.BUCKETS) and len(cs) <= 64
    assert all(c == {"numtokens": c["numtokens"], "heads": 8, "topk": 2048} for c in cs)
    assert D.cases({**GLM, "architecture": "DeepseekV32ForCausalLM", "heads": 16})[0]["heads"] == 16
    assert D.cases({**GLM, "architecture": "Qwen3ForCausalLM"}) == []
    assert D.cases({**GLM, "heads": None}) == []
    k, = D.KERNELS
    assert re.fullmatch(r"[a-z0-9_]{1,40}", k["name"]) and k["name"] == D.KERNEL
    for key in ("body", "cases", "make", "reference", "baseline", "check"):
        assert callable(k[key]), key
    assert k["baseline_name"] == "oxide nvfp4_ds_mla"
    # replaces = exactly the oxide reader entries (the writer stays on oxide)
    src = (REPO / "kernels-oxide/nvfp4_ds_mla/src/main.rs").read_text()
    entries = re.findall(r"pub fn (nvfp4_ds_mla_\w+)\(", src)
    hit = {e for e in entries if any(re.search(r, e, re.I) for r in k["replaces"])}
    assert hit == {"nvfp4_ds_mla_attn_partial", "nvfp4_ds_mla_attn_merge"}, entries


def test_check_is_the_reader_gate():
    ref = torch.randn(3, 2, D.DIM)
    ref[1] = 0  # all -1 token
    out = ref.bfloat16().clone()
    assert D.check(out, ref)[0]
    bad = out.clone()
    bad[1, 0, 0] = 1e-3
    assert not D.check(bad, ref)[0]  # masked row must be exactly 0
    bad = out.clone()
    bad[0, 1, 3] = float("nan")
    assert not D.check(bad, ref)[0]
    assert not D.check(out * 1.05, ref)[0]  # 5e-2 rel-L2 > 2e-2


def test_views_flat_byte_geometry():
    q, cache, topk = problem()
    q3, flat, tk, o3, bs, stride = args_of(q, cache, topk)[:6]
    assert (bs, stride) == (16, 16 * D.ROW + 32)
    assert flat.numel() == 5 * stride + 16 * D.ROW and flat.data_ptr() == cache.data_ptr()
    for s in (0, 15, 16, 95):
        off = (s // bs) * stride + (s % bs) * D.ROW
        assert torch.equal(flat[off:off + D.ROW], cache[s // bs, s % bs])
    # the wrapper's singleton-axis views ([T, 1, ...]) are accepted
    v = D.views(q.unsqueeze(1), cache, topk.unsqueeze(1),
                torch.empty(4, 1, 4, D.DIM, dtype=torch.bfloat16))
    assert v[0].shape == (4, 4, 576) and v[2].shape == (4, 40) and v[3].shape == (4, 4, 512)
    out = torch.empty(4, 4, D.DIM, dtype=torch.bfloat16)
    for bad in ((q.half(), cache, topk, out), (q, cache, topk.long(), out),
                (q, cache, topk, out[:, :2]), (q, cache[:, :, :300], topk, out),
                (q, torch.zeros(3 * (8 * 352 + 8), dtype=torch.uint8).as_strided(
                    (3, 8, 352), (8 * 352 + 8, 352, 1)), topk, out)):  # 8 B block stride
        with pytest.raises(RuntimeError, match="ds_mla"):
            D.views(*bad)


@pytest.fixture
def knob(monkeypatch):
    """Configs shipped for `tuned` buckets; compiled() -> recording fakes."""
    calls, verified = [], []
    tuned = set()
    monkeypatch.setattr(hk, "config", lambda k, c, plat=None: (
        {"num_splits": 4} if k == D.KERNEL and c["numtokens"] in tuned else None))
    monkeypatch.setattr(hk, "platform", lambda device=0: "gpu")

    def compiled(body, kernel, case, **kw):
        assert body is D.ds_mla_decode and kernel == D.KERNEL
        return lambda *a: calls.append((case["numtokens"], a))
    monkeypatch.setattr(hk, "compiled", compiled)
    monkeypatch.setattr(D, "_verify", lambda route, dev: verified.append(sorted(route["kernels"])))
    monkeypatch.setattr(D, "_VERIFIED", set())
    monkeypatch.setattr(D, "helion", D.helion or object())
    monkeypatch.delenv(D.KNOB, raising=False)
    return types.SimpleNamespace(calls=calls, verified=verified, tuned=tuned, mp=monkeypatch)


def test_knob_oxide_helion_auto(knob):
    assert D.load(4, 40, "cpu") is None  # default oxide: nothing compiled
    knob.mp.setenv(D.KNOB, "auto")
    assert D.load(4, 40, "cpu") is None and knob.verified == []  # nothing tuned
    knob.tuned.update({1, 8})
    knob.mp.setattr(D, "_VERIFIED", set())
    route = D.load(4, 40, "cpu")
    assert sorted(route["kernels"]) == [1, 8] and knob.verified == [[1, 8]]
    D.load(4, 40, "cpu")
    assert knob.verified == [[1, 8]]  # once per device, not per layer
    q, cache, topk = problem()
    out = torch.empty(4, 4, D.DIM, dtype=torch.bfloat16)
    assert D.decode(route, q[:1], cache, topk[:1], out[:1], 0.5) is not None
    (b, a), = knob.calls
    assert b == 1 and a[0].shape == (1, 4, 576) and a[4:] == (16, 16 * D.ROW + 32, 0.5)
    assert D.decode(route, q, cache, topk, out, 0.5) is None  # T=4 -> bucket 4: untuned
    assert D.decode(route, q[:1, :2], cache, topk[:1], out[:1, :2], 0.5) is None  # heads
    knob.mp.setenv(D.KNOB, "helion")
    with pytest.raises(RuntimeError, match="no tuned ds_mla_decode config"):
        D.load(4, 40, "cpu")
    knob.tuned.update(D.BUCKETS)
    route = D.load(4, 40, "cpu")
    assert sorted(route["kernels"]) == list(D.BUCKETS)
    assert D.decode(route, q, cache, topk, out, 0.5) is out and knob.calls[-1][0] == 4
    with pytest.raises(RuntimeError, match="helion: no ds_mla_decode kernel"):
        D.decode(route, q, cache, topk[:, :20], out, 0.5)
    knob.mp.setenv(D.KNOB, "triton")
    with pytest.raises(RuntimeError, match="expected oxide"):
        D.load(4, 40, "cpu")


def test_patch_dispatch_and_auto_fallback(knob, monkeypatch):
    oxide = []
    monkeypatch.setattr(P, "_native", lambda: types.SimpleNamespace(
        nvfp4_ds_mla_plan=lambda t, h, c, sms: {"ns": 3},
        nvfp4_ds_mla_decode_cuda=lambda *a: oxide.append(a)))
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda d=None: types.SimpleNamespace(cuda_stream=42))
    impl = types.SimpleNamespace(scale=0.0625, _nvfp4_num_sms=188, num_heads=4,
                                 topk_indices_buffer=torch.zeros(9, 40, dtype=torch.int32))
    assert P._helion_route(impl, 0) is None  # knob unset: oxide, port untouched
    knob.mp.setenv(D.KNOB, "auto")
    knob.tuned.add(1)
    impl._suffix_helion_ds_mla = P._helion_route(impl, 0)
    assert sorted(impl._suffix_helion_ds_mla["kernels"]) == [1]
    q, cache, topk = problem()
    out = torch.empty(4, 4, D.DIM, dtype=torch.bfloat16)
    assert P._suffix_nvfp4_ds_mla_decode(impl, q[:1], cache, topk[:1], out[:1]) is not None
    assert len(knob.calls) == 1 and oxide == []
    assert P._suffix_nvfp4_ds_mla_decode(impl, q, cache, topk, out) is out  # T=4: oxide
    a, = oxide
    assert a[3] is out and a[4].shape == (4, 4, 3, 512) and a[6] == 0.0625
    impl._suffix_helion_ds_mla = None  # oxide route: never touches the port
    monkeypatch.setattr(D, "decode", lambda *a: pytest.fail("Helion called"))
    P._suffix_nvfp4_ds_mla_decode(impl, q, cache, topk, out)
    assert len(oxide) == 2


def test_load_time_oracle_accepts_exact_rejects_nan():
    def exact(q3, flat, tk, o3, bs, stride, scale):
        o3.copy_(D.reference(q3, flat, tk, o3, bs, stride, scale))

    def stale(q3, flat, tk, o3, bs, stride, scale):
        o3.fill_(float("nan"))
    route = {"mode": "auto", "heads": 2, "topk": 40, "kernels": {1: exact, 6: exact}}
    torch.set_default_dtype(torch.bfloat16)  # as under vLLM model init
    try:
        D._verify(route, torch.device("cpu"))
        assert torch.get_default_dtype() == torch.bfloat16
        route["kernels"][6] = stale
        with pytest.raises(RuntimeError, match="load-time oracle FAILED at T=6"):
            D._verify(route, torch.device("cpu"))
    finally:
        torch.set_default_dtype(torch.float32)


# ---- Helion eager reference mode (HELION_INTERPRET=1) ----------------------
@pytest.fixture
def interp(monkeypatch):
    helion = pytest.importorskip("helion")
    if D.helion is None:
        pytest.skip("ds_mla imported before helion was installed")
    real = torch.mm

    def mm(a, b, *args, out_dtype=None, **kw):
        # CPU torch has no mm(out_dtype=): f32 math is the same exact-product,
        # f32-accumulate semantics hl.dot has on the GPU
        if out_dtype is None or a.is_cuda:
            return real(a, b, *args, **kw) if out_dtype is None else real(
                a, b, *args, out_dtype=out_dtype, **kw)
        return real(a.float(), b.float()).to(out_dtype)
    monkeypatch.setattr(torch, "mm", mm)
    return helion


@pytest.mark.parametrize("num_splits", [1, 4, 64])  # 1 / 3 (16,16,8) / 40 of 64 splits
def test_body_matches_oracle_reference(interp, num_splits):
    q, cache, topk = problem(seed=num_splits)
    a = args_of(q, cache, topk)
    kern = interp.kernel(D.ds_mla_decode, config=interp.Config(num_splits=num_splits))
    out = kern(*a)
    assert out is a[3]
    ok, msg = D.check(out, D.reference(*a))
    assert ok, msg
    assert bool((out[1] == 0).all())  # all -1 token


def test_load_time_oracle_runs_the_body(interp):
    kern = interp.kernel(D.ds_mla_decode, config=interp.Config(num_splits=4))
    route = {"mode": "helion", "heads": 2, "topk": 40, "kernels": {1: kern, 2: kern, 6: kern}}
    D._verify(route, torch.device("cpu"))
