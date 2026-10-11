# SPDX-License-Identifier: Apache-2.0
"""CPU tests for nvfp4_ds_mla_patch.ours (our Triton sparse-MLA decode): the
plan, the 352 B row dequant in the kernel's f16 domain, realistic list
generation, the union rule and per-row masks (torch emulation of the kernel
algorithm vs the f64 reference), mode / table selection, the patch routing,
and — when triton is importable — the Triton kernel itself in the interpreter."""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "sm120"))
import nvfp4_ds_mla_patch as P  # noqa: E402
from nvfp4_ds_mla_patch import oracle as O  # noqa: E402
from nvfp4_ds_mla_patch import ours  # noqa: E402

SMS = 188


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ours.MODE_ENV, raising=False)
    monkeypatch.delenv(ours.TABLE_ENV, raising=False)
    monkeypatch.setattr(ours, "_MODE", "oxide")
    monkeypatch.setattr(ours, "_TABLE", None)
    monkeypatch.setattr(ours, "CFG", {"asm": True})
    ours._AUTO.clear()


# ---- plan --------------------------------------------------------------------
@pytest.mark.parametrize("hq", [8, 16, 64])
@pytest.mark.parametrize("c", [2048, 1000, 64, 33])
@pytest.mark.parametrize("cfg", list(ours.CONFIGS))
def test_plan_one_wave_bounded_splits(hq, c, cfg):
    k = ours.CONFIGS[cfg]
    for t in (1, 2, 3, 6, 8, 24, 96, 192, 1024, 8192):
        p = ours.plan(t, hq, c, SMS, cfg)
        assert 1 <= p["ns"] <= min(ours.NS_MAX, -(-c // k["tn"])) and ours.NS_MAX <= ours.NSP
        assert p["ns"] == 1 or t * p["hqt"] * p["ns"] <= SMS * k["occ"]  # one wave
        assert p["hqt"] * p["hb"] >= hq and p["hb"] <= p["m"] == ours.M_ROWS
        assert p["cp"] >= c and p["cp"] & (p["cp"] - 1) == 0 and p["tn"] == k["tn"]
        assert ours.plan(t, hq, c, SMS, cfg) == p  # pure: CUDA-graph stable


def test_plan_glm_shapes():
    got = {(t, cfg): ours.plan(t, 8, 2048, SMS, cfg)["ns"]
           for t in (1, 6, 24, 96, 192) for cfg in ("t32", "t64")}
    assert got == {(1, "t32"): 16, (6, "t32"): 16, (24, "t32"): 15, (96, "t32"): 3,
                   (192, "t32"): 1, (1, "t64"): 16, (6, "t64"): 16, (24, "t64"): 7,
                   (96, "t64"): 1, (192, "t64"): 1}
    assert ours.plan(4, 64, 2048, SMS)["hqt"] == 4
    assert ours.plan(70000, 8, 2048, SMS)["ns"] == 1  # beyond the merge counters
    with pytest.raises(ValueError):
        ours.plan(0, 8, 2048, SMS)


def test_q_group_never_straddles_requests():
    assert {q: ours.q_group(q) for q in range(0, 9)} == {
        0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 3, 7: 1, 8: 4}


# ---- data, dequant, union rule ----------------------------------------------
def _cache(n=4096, seed=0):
    import torch

    gen = torch.Generator().manual_seed(seed)
    kv_c, k_pe = O.make_kv(n, "cpu", gen, special=True)
    rows = O.quant_rows_ref(kv_c, k_pe)
    lat, rope = (x.double() for x in O.dequant_rows_ref(rows))
    return rows.view(n // 64, 64, O.ROW).clone(), lat, rope


def test_f16_rows_is_the_352b_dequant_times_2_pow_minus_8():
    import torch

    cache, lat, rope = _cache()
    rows = cache.view(-1, O.ROW)
    k_lo, k_hi, k_pe = ours._f16_rows(rows)
    lat2 = torch.stack([k_lo, k_hi], -1).view(-1, 512).double() * 256
    assert torch.equal(lat2, lat) and torch.equal(k_pe.double() * 256, rope)
    # every value is an exact f16 (the kernel's operand domain)
    for x in (k_lo, k_hi, k_pe):
        assert torch.equal(x.half().float(), x)


def test_make_lists_short_prefixes_and_long_topk_sets():
    import torch

    gen = torch.Generator().manual_seed(1)
    s = ours.make_lists(3, 6, 1 << 14, gen, "short")
    assert s.shape == (18, 2048)
    for r in range(3):
        blk = s[6 * r:6 * r + 6]
        lens = [ours.live_len(x) for x in blk]
        assert lens == list(range(lens[0], lens[0] + 6))       # causal: +1 per token
        assert all(bool((x[:n] >= 0).all() and (x[n:] < 0).all()) for x, n in zip(blk, lens))
        assert ours.union_ok(s, 6 * r, 6 * r + 5)
        assert len(set(blk[-1][:lens[-1]].tolist())) == lens[-1]  # distinct slots
    lg = ours.make_lists(2, 3, 1 << 14, gen, "long")
    assert bool((lg >= 0).all()) and not ours.union_ok(lg, 0, 2)
    a, b = set(lg[0].tolist()), set(lg[1].tolist())
    assert len(a) == 2048 and 0.75 < len(a & b) / 2048 < 0.95  # neighbors share most rows


def test_union_rule_edge_cases():
    import torch

    ref = torch.tensor([[5, 6, 7, -1]])
    ok = torch.cat([torch.tensor([[5, -1, 7, -1], [-1, -1, -1, -1]]), ref])
    assert ours.union_ok(ok, 0, 2) and ours.union_ok(ref, 0, 0)
    extra = torch.cat([torch.tensor([[5, 6, 7, 9]]), ref])    # a slot the ref lacks
    moved = torch.cat([torch.tensor([[6, 5, -1, -1]]), ref])  # same set, other columns
    assert not ours.union_ok(extra, 0, 1) and not ours.union_ok(moved, 0, 1)


@pytest.mark.parametrize("q_len,n_req,regime,hq,c", [
    (6, 1, "short", 8, 2048), (1, 2, "short", 8, 2048), (3, 4, "short", 8, 2048),
    (4, 2, "short", 8, 2048), (6, 1, "long", 8, 2048), (3, 2, "long", 8, 1000),
    (1, 2, "short", 64, 2048), (2, 2, "long", 64, 2048),
])
@pytest.mark.parametrize("cfg", ["t32", "t64"])
def test_emulation_matches_f64_reference(q_len, n_req, regime, hq, c, cfg):
    import torch

    cache, lat, rope = _cache()
    gen = torch.Generator().manual_seed(q_len * 7 + n_req)
    topk = ours.make_lists(n_req, q_len, 4096, gen, regime, topk=c, ctx=(40, 300))
    t = topk.shape[0]
    if t > 2:
        topk[1, 3::5] = -1  # holes
        topk[-2] = -1       # an all -1 token
    q = (torch.randn(t, hq, 576, generator=gen) * 2).bfloat16()
    out = ours.emulate(q, cache, topk, O.SCALE, SMS, cfg)
    ref = O.attention_ref(q.double(), lat, rope, topk, O.SCALE)
    valid = (topk >= 0).any(-1)
    keep = valid[:, None].expand(t, hq)
    assert O.row_rel_l2(out, ref, keep) <= 1e-2
    assert bool((out[~valid] == 0).all()) and bool(torch.isfinite(out).all())


def test_emulation_direct_and_split_paths_agree():
    import torch

    cache, lat, rope = _cache()
    gen = torch.Generator().manual_seed(5)
    topk = ours.make_lists(2, 3, 4096, gen, "short", ctx=(100, 300))
    q = torch.randn(6, 8, 576, generator=gen).bfloat16()
    a = ours.emulate(q, cache, topk, O.SCALE, 1)      # 1 SM: ns 1, no merge
    b = ours.emulate(q, cache, topk, O.SCALE, SMS)    # ns 16 + merge
    assert ours.plan(6, 8, 2048, 1)["ns"] == 1 and ours.plan(6, 8, 2048, SMS)["ns"] == 16
    assert O.rel_l2(a, b) <= 5e-3


# ---- mode / table ------------------------------------------------------------
def _table(tmp_path, monkeypatch, tbl):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(tbl))
    monkeypatch.setenv(ours.TABLE_ENV, str(path))
    return path


ROWS = [{"T": 6, "ours": True, "cfg": "t32"}, {"T": 96, "ours": False, "cfg": "t64"},
        {"T": 1, "ours": True, "cfg": "t32s1"}]


def test_mode_oxide_default_and_bad_value(monkeypatch):
    assert ours.init() == "oxide" and not ours.use_ours(6, 3)
    monkeypatch.setenv(ours.MODE_ENV, "fast")
    with pytest.raises(ValueError):
        ours.init()


def test_mode_ours_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv(ours.MODE_ENV, "ours")
    monkeypatch.setenv(ours.TABLE_ENV, str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="no oracle table"):
        ours.init()
    _table(tmp_path, monkeypatch, {"pass": False, "rows": [], "failed": ["x"]})
    with pytest.raises(RuntimeError, match="FAILED"):
        ours.init()
    _table(tmp_path, monkeypatch, {"pass": True, "rows": ROWS, "asm": False})
    assert "oracle pass" in ours.init() and ours.CFG == {"asm": False}
    assert ours.use_ours(96, 3) and ours.use_ours(5, 2)  # ours: everywhere


def test_mode_auto_follows_the_table(monkeypatch, tmp_path):
    monkeypatch.setenv(ours.MODE_ENV, "auto")
    monkeypatch.setenv(ours.TABLE_ENV, str(tmp_path / "missing.json"))
    assert "oxide everywhere" in ours.init() and not ours.use_ours(6, 3)
    _table(tmp_path, monkeypatch, {"pass": True, "rows": ROWS})
    assert "1/3" not in ours.init()
    assert ours.use_ours(6, 3) and ours.use_ours(12, 3)          # nearest T: 6
    assert not ours.use_ours(96, 3) and not ours.use_ours(48, 3)  # nearest T: 96
    assert ours.use_ours(1, 1) and ours.use_ours(2)               # nearest T: 1 (then 6)
    assert [ours.config_for(t) for t in (1, 2, 6, 24, 96, 8192)] == [
        "t32s1", "t32s1", "t32", "t64", "t64", "t64"]
    _table(tmp_path, monkeypatch, {"pass": False, "rows": ROWS})
    assert "oxide everywhere" in ours.init() and not ours.use_ours(6, 3)


def test_config_for_defaults_without_a_table(monkeypatch):
    assert ours.config_for(6) == ours.DEFAULT
    monkeypatch.setattr(ours, "_TABLE", {"pass": True, "rows": [{"T": 6, "cfg": "gone"}]})
    assert ours.config_for(6) == ours.DEFAULT  # a config this build no longer has


def test_use_asm(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    assert ours.use_asm()
    ours.CFG["asm"] = False
    assert not ours.use_asm()
    ours.CFG["asm"] = True
    monkeypatch.setenv("TRITON_INTERPRET", "1")  # the interpreter has no inline asm
    assert not ours.use_asm()


# ---- patch routing -----------------------------------------------------------
def test_q_group_hook():
    ns = types.SimpleNamespace
    assert P._suffix_dsmla_q_group(ns(num_decodes=8, num_decode_tokens=24)) == (3, 24)
    assert P._suffix_dsmla_q_group(ns(num_decodes=1, num_decode_tokens=6)) == (3, 6)
    assert P._suffix_dsmla_q_group(ns(num_decodes=3, num_decode_tokens=7)) == (1, 7)
    assert P._suffix_dsmla_q_group(ns(num_decodes=0, num_decode_tokens=0)) == (1, -1)
    assert P._suffix_dsmla_q_group(object()) == (1, -1)


def test_forward_mqa_records_the_group_in_both_pins():
    for d in ("vllm_0.30.0", "vllm_0.31.1rc1.dev273"):
        src = (REPO / "sm120/tests/fixtures" / d / "flashinfer_mla_sparse_sm120.py").read_text()
        new, _ = P.patch_source(P.IMPL_MODULE, src)
        fwd = new[new.index("    def forward_mqa("):new.index("    def _run_mqa_kernel(")]
        assert "self._suffix_q_group = _suffix_dsmla_q_group(attn_metadata)" in fwd


def test_decode_helper_routes_by_mode(monkeypatch):
    import torch

    calls = []
    native = types.SimpleNamespace(
        HAS_NVFP4_DSMLA_CUDA=True, nvfp4_ds_mla_plan=lambda t, h, c, sms: {"ns": 2},
        nvfp4_ds_mla_decode_cuda=lambda *a: calls.append("oxide"))
    pkg = types.ModuleType("suffix_hybrid")
    pkg._native = native
    monkeypatch.setitem(sys.modules, "suffix_hybrid", pkg)
    monkeypatch.setitem(sys.modules, "suffix_hybrid._native", native)
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda d=None: types.SimpleNamespace(cuda_stream=1))
    monkeypatch.setattr(ours, "decode", lambda q, kv, tk, out, scale, g, sms:
                        calls.append(("ours", g)) or out)
    impl = types.SimpleNamespace(scale=0.0625, _nvfp4_num_sms=SMS, _suffix_q_group=(3, 24))
    q = torch.zeros(24, 8, 576, dtype=torch.bfloat16)
    out = torch.empty(24, 8, 512, dtype=torch.bfloat16)
    kv, topk = torch.zeros(4, 64, 352, dtype=torch.uint8), torch.zeros(24, 2048, dtype=torch.int32)
    P._suffix_nvfp4_ds_mla_decode(impl, q, kv, topk, out)            # mode oxide
    monkeypatch.setattr(ours, "_MODE", "ours")
    monkeypatch.setattr(ours, "_TABLE", {"pass": True, "rows": []})
    P._suffix_nvfp4_ds_mla_decode(impl, q, kv, topk, out)            # decode slice: G 3
    P._suffix_nvfp4_ds_mla_decode(impl, q[:5], kv, topk[:5], out[:5])  # prefill slice: G 4
    monkeypatch.setattr(ours, "_TABLE", {"pass": True, "rows": [{"T": 24, "G": 3, "ours": False}]})
    monkeypatch.setattr(ours, "_MODE", "auto")
    P._suffix_nvfp4_ds_mla_decode(impl, q, kv, topk, out)            # auto: table says oxide
    assert calls == ["oxide", ("ours", 3), ("ours", 4), "oxide"]


# ---- sitecustomize -----------------------------------------------------------
def _site(tmp_path, mode, oracle_rc, oracle_pass):
    pkg = tmp_path / "nvfp4_ds_mla_patch"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "ours.py").write_text(
        "import json, os, sys\n"
        f"open({str(tmp_path / 'seen.json')!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], "
        "'env': sorted(k for k in os.environ if k.startswith('SUFFIX_'))}))\n"
        f"json.dump({{'pass': {oracle_pass}, 'rows': []}}, open(sys.argv[sys.argv.index('--table') + 1], 'w'))\n"
        f"sys.exit({oracle_rc})\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    env.update(PYTHONPATH=f"{REPO}{os.pathsep}{tmp_path}", SUFFIX_SM120_NVP4DSMLA="1",
               SUFFIX_SM120_DSMLA_ATTN=mode, SUFFIX_SM120_DSMLA_TABLE=str(tmp_path / "tbl.json"))
    return subprocess.run([sys.executable, "-c", "import sitecustomize"], env=env,
                          capture_output=True, text=True, cwd=tmp_path)


def test_sitecustomize_runs_the_oracle_once_and_fails_closed_for_ours(tmp_path):
    r = _site(tmp_path, "ours", 0, True)
    assert r.returncode == 0 and "[suffix dsmla-ours] oracle PASS" in r.stderr, r.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["argv"][:1] == ["oracle"] and "SUFFIX_SM120_DSMLA_ATTN" not in seen["env"]
    (tmp_path / "seen.json").unlink()
    r = _site(tmp_path, "ours", 0, True)  # second top-level process: no rerun
    assert r.returncode == 0 and not (tmp_path / "seen.json").exists()
    for f in ("tbl.json", "tbl.json.claimed"):
        (tmp_path / f).unlink()
    r = _site(tmp_path, "ours", 1, False)
    assert r.returncode != 0 and "refusing to serve" in r.stderr
    for f in ("tbl.json", "tbl.json.claimed"):
        (tmp_path / f).unlink()
    r = _site(tmp_path, "auto", 1, False)
    assert r.returncode == 0 and "oracle FAIL" in r.stderr


def test_sitecustomize_oracle_inert_without_mode(tmp_path):
    r = _site(tmp_path, "oxide", 0, True)
    assert r.returncode == 0 and not (tmp_path / "seen.json").exists()


def test_dsmla_attn_bench_gate_runs_ours_bench():
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    argv, env = ns["_BOOT_GATES"]["dsmla_attn_bench"]
    assert argv[:3] == ["-m", "nvfp4_ds_mla_patch.ours", "bench"] and env == {}


# ---- the Triton kernel itself (interpreter; skipped without triton) ----------
INTERP = r"""
import os, sys
import torch
sys.path.insert(0, sys.argv[1])
from nvfp4_ds_mla_patch import oracle as O, ours
gen = torch.Generator().manual_seed(0)
kv_c, k_pe = O.make_kv(4096, "cpu", gen, special=True)
rows = O.quant_rows_ref(kv_c, k_pe)
cache = rows.view(64, 64, O.ROW).clone()
lat, rope = (x.double() for x in O.dequant_rows_ref(rows))
for q_len, n_req, regime, hq, sms, cfg in [(6, 1, "short", 8, 188, "t32"), (3, 2, "long", 8, 188, "t64"),
                                           (1, 2, "short", 64, 188, "t32"), (4, 2, "short", 8, 1, "t64")]:
    topk = ours.make_lists(n_req, q_len, 4096, gen, regime, ctx=(40, 200))
    t = topk.shape[0]
    if t > 2:
        topk[1, 3::5] = -1
        topk[-2] = -1
    q = (torch.randn(t, hq, 576, generator=gen) * 2).bfloat16()
    out = ours.decode(q, cache, topk, torch.empty(t, hq, 512, dtype=torch.bfloat16),
                      O.SCALE, 1, sms, cfg=cfg)
    out2 = ours.decode(q, cache, topk, torch.empty(t, hq, 512, dtype=torch.bfloat16),
                       O.SCALE, 1, sms, cfg=cfg)  # the merge counters came back rearmed
    emu = ours.emulate(q, cache, topk, O.SCALE, sms, cfg)
    ref = O.attention_ref(q.double(), lat, rope, topk, O.SCALE)
    valid = (topk >= 0).any(-1)
    e_ref = O.row_rel_l2(out, ref, valid[:, None].expand(t, hq))
    e_emu = O.rel_l2(out[valid], emu[valid])
    assert torch.equal(out, out2) and bool((ours._SEMS[-1] == 0).all())
    assert bool((out[~valid] == 0).all()) and e_ref <= 2e-2 and e_emu <= 1.5e-2, (e_ref, e_emu)
    print("ok", q_len, n_req, regime, hq, sms, cfg, ours.plan(t, hq, 2048, sms, cfg)["ns"],
          f"{e_ref:.2e} {e_emu:.2e}")
"""


@pytest.mark.skipif(importlib.util.find_spec("triton") is None and not os.environ.get(
    "SUFFIX_TEST_TRITON_PATH"), reason="triton not installed (interpreter test)")
def test_triton_kernel_in_interpreter():
    env = dict(os.environ, TRITON_INTERPRET="1")
    extra = os.environ.get("SUFFIX_TEST_TRITON_PATH")  # a triton checkout (local runs)
    if extra:
        env["PYTHONPATH"] = os.pathsep.join([extra, env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, "-c", INTERP, str(REPO / "sm120")], env=env,
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr[-3000:]
    assert r.stdout.count("ok ") == 4, r.stdout


# ---- oracle / bench harness dry run (kernels replaced by exact references) ---
class _Graph:
    capturing = None

    def __init__(self):
        self.fn = None

    def replay(self):
        self.fn()


def _captured(fn):
    if _Graph.capturing is not None:
        _Graph.capturing.fn = fn
    else:
        fn()


def _ref_into(out, q, cache, topk):
    lat, rope = O.dequant_rows_ref(cache.reshape(-1, O.ROW))
    out.copy_(O.attention_ref(q.float(), lat, rope, topk, O.SCALE))


def _fake_decode(q, cache, topk, out, scale, group, sms, asm=None, cfg=None):
    ours.plan(q.shape[0], q.shape[1], topk.shape[1], sms, cfg or ours.DEFAULT)
    _captured(lambda: _ref_into(out, q, cache, topk))
    return out


class _FakeBase:
    P = P
    impl = types.SimpleNamespace(scale=O.SCALE, _nvfp4_num_sms=SMS)

    def write(self, kv_c, k_pe, cache, slots):
        rows = O.quant_rows_ref(kv_c, k_pe)
        cache.view(-1, O.ROW)[slots[:rows.shape[0]]] = rows


class _FakeStock:
    def write(self, kv_c, k_pe, cache, slots):
        import torch

        x = kv_c.float().view(-1, 4, 128)
        scale = torch.clamp(x.abs().amax(-1) / 448.0, min=1e-30)
        rows = torch.zeros(kv_c.shape[0], O.FP8_ROW, dtype=torch.uint8)
        rows[:, :512] = O.e4m3_bytes((x / scale[..., None]).view(-1, 512))
        rows[:, 512:528] = scale.contiguous().view(torch.uint8)
        rows[:, 528:] = k_pe.contiguous().view(torch.uint8)
        cache.view(-1, O.FP8_ROW)[slots] = rows

    def decode(self, q, cache, topk, out=None):
        out = q.new_empty(q.shape[0], q.shape[1], O.DIM) if out is None else out
        lat, rope = O.dequant_fp8_rows(cache.reshape(-1, O.FP8_ROW))
        _captured(lambda: out.copy_(O.attention_ref(q.float(), lat, rope, topk, O.SCALE)))
        return out


def test_oracle_and_bench_harness_dry_run_on_cpu(monkeypatch, tmp_path):
    """run_oracle / run_bench / main end to end with exact references in place
    of the GPU kernels: a silicon run can only fail on kernel numerics."""
    _harness(monkeypatch, tmp_path)


def test_helpers_and_harness_under_bf16_default_dtype(monkeypatch, tmp_path):
    """vLLM builds models (impl init, the boot self-test) under
    torch.set_default_dtype(bfloat16): no helper may depend on the default."""
    import torch

    cache, lat, rope = _cache()
    gen = torch.Generator().manual_seed(9)
    topk = ours.make_lists(2, 3, 4096, gen, "short", ctx=(40, 200))
    q = (torch.randn(6, 8, 576, generator=gen, dtype=torch.float32) * 2).bfloat16()
    ref = O.attention_ref(q.double(), lat, rope, topk, O.SCALE)
    emu = ours.emulate(q, cache, topk, O.SCALE, SMS)
    rows16 = ours._f16_rows(cache.view(-1, O.ROW))
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        cache_b, lat_b, rope_b = _cache()
        assert torch.equal(cache_b, cache) and torch.equal(lat_b, lat)
        assert torch.equal(rope_b, rope)
        gen_b = torch.Generator().manual_seed(9)
        assert torch.equal(ours.make_lists(2, 3, 4096, gen_b, "short", ctx=(40, 200)), topk)
        assert torch.equal(O.attention_ref(q.double(), lat, rope, topk, O.SCALE), ref)
        assert torch.equal(ours.emulate(q, cache, topk, O.SCALE, SMS), emu)
        assert all(torch.equal(a, b) for a, b in zip(ours._f16_rows(cache.view(-1, O.ROW)),
                                                      rows16))
        assert ours.init() == "oxide"
        _harness(monkeypatch, tmp_path)
    finally:
        torch.set_default_dtype(prev)


def _harness(monkeypatch, tmp_path):
    import contextlib

    import torch

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    stream = types.SimpleNamespace(wait_stream=lambda s: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    monkeypatch.setattr(torch.cuda, "Event", lambda *a: types.SimpleNamespace(
        record=lambda: None, elapsed_time=lambda other: 0.005))

    @contextlib.contextmanager
    def graph(g):
        _Graph.capturing = g
        yield
        _Graph.capturing = None
    monkeypatch.setattr(torch.cuda, "graph", graph)
    monkeypatch.setattr(ours, "decode", _fake_decode)
    monkeypatch.setattr(ours, "_oxide", lambda base, q, cache, topk, out: (
        _captured(lambda: _ref_into(out, q, cache, topk)) or out))
    monkeypatch.setattr(O, "Stock", lambda dev: _FakeStock())
    monkeypatch.setattr(O, "PREFILL_T", 96)
    monkeypatch.setattr(ours, "ORACLE_SHAPES", [(6, 1), (3, 4), (1, 3)])
    monkeypatch.setattr(ours, "BENCH_SHAPES", [(1, 2), (6, 1), (3, 4)])
    monkeypatch.setattr(ours, "FLUSH_BYTES", 1024)
    results = []

    def report(name, ok, detail):
        results.append((name, ok, detail))

    configs = ours.run_oracle(_FakeBase(), "cpu", report, nslots=1 << 14)
    names = [r[0] for r in results]
    assert all(ok for _n, ok, _d in results), [r for r in results if not r[1]]
    assert configs == list(ours.CONFIGS)
    assert {"ours_short_q6x1", "ours_long_q3x4", "ours_short_q1x3", "reader_cuda_graph_replay",
            "adversarial_prefill_T96"} <= set(names)
    assert {f"ours_cuda_graph_replay_{c}" for c in ours.CONFIGS} <= set(names)
    assert names.count("reader_padded_block_stride") == 1
    assert ours.CFG == {"asm": True}
    rows = ours.run_bench(_FakeBase(), "cpu", reps=2, nslots=1 << 14, configs=configs)
    assert [r["T"] for r in rows] == [2, 6, 12]
    assert all(r["ours_short"] == 5.0 and r["ours"] and r["cfg"] == ours.DEFAULT for r in rows)
    # main(): verdict -> table -> exit code
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(O, "Ours", lambda idx: _FakeBase())
    monkeypatch.setattr(ours, "run_oracle", lambda b, d, rep: rep("x", True, "ok") or ["t32"])
    monkeypatch.setattr(ours, "run_bench", lambda b, d, reps, configs=None: rows)
    path = tmp_path / "t.json"
    assert ours.main(["oracle", "--table", str(path)]) == 0
    tbl = json.loads(path.read_text())
    assert tbl["pass"] and tbl["rows"] == rows and tbl["configs"] == ["t32"]
    assert tbl["failed"] == [] and "kernels" in tbl
    monkeypatch.setattr(ours, "run_oracle", lambda b, d, rep: rep("y", False, "bad"))
    assert ours.main(["oracle", "--table", str(path)]) == 1
    assert ours.main(["bench", "--table", str(path)]) == 0
    tbl = json.loads(path.read_text())
    assert not tbl["pass"] and tbl["failed"] == ["y"] and tbl["rows"] == []
