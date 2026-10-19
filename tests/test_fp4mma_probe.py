# SPDX-License-Identifier: Apache-2.0
"""K5 plumbing, the stack gate, the PTX judge, the capture/replay machinery, the P0 script and the
bake-off harness logic (kernels/fp4mma_probe.py, scripts/fp4mma_p0_probe.py, kernels/fp4mma_bench.py):
everything that does NOT need Triton or a GPU. The real-Triton compile of the mirror kernels is
tests/test_fp4mma_ptx.py (Linux CI only)."""
import importlib.util
import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from suffix_hybrid.kernels import fp4mma_probe as probe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NATIVE_PTX = f""".version 9.1
.target sm_120a
.address_size 64
.visible .entry _helion_nvfp4_gemm_q_fp4mma(
    .param .u64 p0
)
{{
    {probe.NATIVE_MMA} {{%f1, %f2, %f3, %f4}}, {{%r1, %r2, %r3, %r4}}, {{%r5, %r6}}, {{%f1, %f2, %f3, %f4}}, {{%r7}}, {{0, 0}}, {{%r8}}, {{0, 0}};
    ret;
}}
"""
DECOMPOSED_PTX = """.version 9.1
.target sm_120a
.visible .entry k()
{
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%f1, %f2, %f3, %f4}, {%r1, %r2, %r3, %r4}, {%r5, %r6}, {%f1, %f2, %f3, %f4};
    ret;
}
"""
MX_PTX = NATIVE_PTX.replace("scale_vec::4X", "scale_vec::2X").replace("ue4m3", "ue8m0")
SPILL_PTX = NATIVE_PTX.replace("{\n    mma", "{\n    .local .align 16 .b8 __local_depot0[256];\n    mma")


def test_native_ptx_passes_and_every_decomposition_is_named():
    info = probe.analyze_ptx(NATIVE_PTX)
    assert info["target"] == "sm_120a" and info["native_mma"] == 1 and not info["local"]
    assert probe.ptx_problems(info, allow_local=False) == []
    bad = probe.ptx_problems(probe.analyze_ptx(DECOMPOSED_PTX))
    assert len(bad) == 1 and "decomposed" in bad[0] and "m16n8k16.row.col.f32.bf16.bf16.f32 x1" in bad[0]
    mx = probe.ptx_problems(probe.analyze_ptx(MX_PTX))
    assert any("ue8m0" in w or "MXFP4" in w for w in mx)
    for tgt in ("sm_90a", "sm_100a", "sm_120"):
        w = probe.ptx_problems(probe.analyze_ptx(NATIVE_PTX.replace("sm_120a", tgt)))
        assert w and "target" in w[0], (tgt, w)
    assert probe.ptx_problems(probe.analyze_ptx(NATIVE_PTX.replace("sm_120a", "sm_120f"))) == []
    # spills: a warning in serving (allow_local), an error in the lab gate and the P0 probe
    spill = probe.analyze_ptx(SPILL_PTX)
    assert spill["local"] and probe.ptx_problems(spill, allow_local=True) == []
    assert "local memory" in probe.ptx_problems(spill, allow_local=False)[0]
    assert probe.ptx_problems(probe.analyze_ptx(""), want_native=False) != []  # no target at all


def test_parse_ptxas_v_on_real_ptxas_text():
    text = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_helion_k' for 'sm_120a'
ptxas info    : Function properties for _helion_k
    64 bytes stack frame, 52 bytes spill stores, 48 bytes spill loads
ptxas info    : Used 168 registers, used 1 barriers, 32768 bytes smem
ptxas info    : Compile time = 150.2 ms
"""
    assert probe.parse_ptxas_v(text) == {"regs": 168, "smem": 32768, "spill_stores": 52,
                                         "spill_loads": 48, "stack": 64}
    assert probe.parse_ptxas_v("ptxas info : Used 40 registers, 380 bytes cmem[0]")["smem"] == 0


def test_parse_version():
    assert probe.parse_version("3.7.1") == (3, 7, 1)
    assert probe.parse_version("3.8.0+git2a3b") == (3, 8, 0)
    assert probe.parse_version("3.6.0rc1") == (3, 6, 0) and probe.parse_version("3.6") == (3, 6, 0)
    assert probe.parse_version(None) is None and probe.parse_version("x") is None


def test_stack_gate_names_every_reason():
    ok = probe.stack_problems((12, 0), "3.8.0", True)
    assert ok == []
    assert probe.stack_problems((12, 1), "3.6.0", True) == []  # RTX 5090 is cc 12.0, PRO 6000 12.0 / 12.1
    why = probe.stack_problems((9, 0), "3.5.1", False)
    assert len(why) == 3
    assert "cc 9.0" in why[0] and "family 120" in why[0]
    assert "3.5.1 < 3.6.0" in why[1] and "triton#8494" in why[1]
    assert "Helion" in why[2]
    assert "no CUDA" in probe.stack_problems(None, "3.8.0", True)[0]
    assert "not installed" in probe.stack_problems((12, 0), None, True)[0]
    # family 100 (tcgen05) is NOT this arm: refused by name, not silently accepted
    assert "cc 10.0" in probe.stack_problems((10, 0), "3.8.0", True)[0]


def test_require_native_stack_raises_with_the_reasons_on_this_cpu_box():
    with pytest.raises(probe.Fp4mmaError, match="cannot run"):
        probe.require_native_stack()  # no CUDA here (and maybe no Triton): fail closed


# ---------------------------------------------------------------------------
# the recorder, driven the way a generated Helion module drives it
# ---------------------------------------------------------------------------
class FakeCK:
    def __init__(self, ptx=NATIVE_PTX, h="h1", spills=0):
        self.asm = {"ptx": ptx}
        self.hash = h
        self.n_regs, self.n_spills = 96, spills
        self.metadata = SimpleNamespace(shared=20480, num_warps=4, num_stages=3)
        self.name = "k"


class FakeJit:
    def __init__(self, name):
        self.fn = SimpleNamespace(__name__=name)


@pytest.fixture
def fake_launcher(monkeypatch):
    hrt = pytest.importorskip("helion.runtime")
    probe.uninstall_recorder()
    probe.clear_records()
    box = {"ck": FakeCK()}
    real = lambda triton_kernel, grid, *a, **kw: box["ck"]
    monkeypatch.setattr(hrt, "default_launcher", real)
    yield box
    probe.uninstall_recorder()
    probe.clear_records()


def test_recorder_records_the_compiled_kernel_the_launcher_returns(fake_launcher):
    import helion.runtime as hrt
    assert probe.install_recorder() is True and probe.install_recorder() is True  # idempotent
    assert probe.recorder_installed() and hrt.default_launcher._real is not hrt.default_launcher
    ck = hrt.default_launcher(FakeJit("_helion_nvfp4_gemm_q_fp4mma"), (1,), 1, 2, num_warps=4, num_stages=3)
    assert ck is fake_launcher["ck"]  # the real return value passes through untouched
    r, = probe.records("_helion_nvfp4")
    assert (r.name, r.n_regs, r.n_spills, r.shared, r.num_warps, r.num_stages) == (
        "_helion_nvfp4_gemm_q_fp4mma", 96, 0, 20480, 4, 3)
    hrt.default_launcher(FakeJit("_helion_nvfp4_gemm_q_fp4mma"), (1,))  # same hash: one record
    assert len(probe.records()) == 1
    lines = []
    got = probe.assert_native(["_helion_nvfp4_gemm_q_fp4mma"], log=lines.append)
    assert len(got) == 1 and lines[0].startswith("K5 ok") and "native_mma=1" in lines[0]
    probe.uninstall_recorder()
    assert not probe.recorder_installed()


def test_k5_fails_closed_on_decomposition_missing_kernel_and_missing_recorder(fake_launcher):
    import helion.runtime as hrt
    with pytest.raises(probe.Fp4mmaError, match="recorder is not installed"):
        probe.assert_native(["x"])
    probe.install_recorder()
    with pytest.raises(probe.Fp4mmaError, match="no compiled kernel recorded for '_helion_dsv41_fp4mma_fc1'"):
        probe.assert_native(["_helion_dsv41_fp4mma_fc1"])
    fake_launcher["ck"] = FakeCK(DECOMPOSED_PTX, "h2")
    hrt.default_launcher(FakeJit("_helion_dsv41_fp4mma_fc1"), (1,))
    with pytest.raises(probe.Fp4mmaError, match="decomposed tl.dot_scaled"):
        probe.assert_native(["_helion_dsv41_fp4mma_fc1"])
    # one bad specialization among good ones still fails (EVERY compile is judged)
    fake_launcher["ck"] = FakeCK(NATIVE_PTX, "h3")
    hrt.default_launcher(FakeJit("_helion_dsv41_fp4mma_fc2"), (1,))
    probe.assert_native(["_helion_dsv41_fp4mma_fc2"])
    fake_launcher["ck"] = FakeCK(MX_PTX, "h4")
    hrt.default_launcher(FakeJit("_helion_dsv41_fp4mma_fc2"), (1,))
    with pytest.raises(probe.Fp4mmaError, match="fc2"):
        probe.assert_native(["_helion_dsv41_fp4mma_fc2"])


def test_spills_warn_in_serving_but_fail_in_the_lab_gate(fake_launcher):
    import helion.runtime as hrt
    probe.install_recorder()
    fake_launcher["ck"] = FakeCK(SPILL_PTX, "s1", spills=64)
    hrt.default_launcher(FakeJit("_helion_nvfp4_gemm_q_fp4mma"), (1,))
    lines = []
    probe.assert_native(["_helion_nvfp4_gemm_q_fp4mma"], allow_local=True, log=lines.append)
    assert any("WARN" in ln and "spilled" in ln for ln in lines)
    with pytest.raises(probe.Fp4mmaError, match="local memory"):
        probe.assert_native(["_helion_nvfp4_gemm_q_fp4mma"], allow_local=False, log=lambda s: None)


def test_recorder_swallows_its_own_errors_and_never_breaks_a_launch(fake_launcher):
    import helion.runtime as hrt
    probe.install_recorder()
    fake_launcher["ck"] = SimpleNamespace(asm=None, hash="bad")  # asm["ptx"] raises inside record_kernel
    assert hrt.default_launcher(FakeJit("k"), (1,)) is fake_launcher["ck"]
    fake_launcher["ck"] = None  # a warm-up / hook run
    assert hrt.default_launcher(FakeJit("k"), (1,)) is None and probe.records() == []


def test_generated_module_binds_the_launcher_at_import_and_the_recorder_sees_the_launch(fake_launcher, monkeypatch, tmp_path):
    """The plumbing claim, executed: Helion's generated module takes `helion.runtime.default_launcher`
    at its first import (so the recorder installed BEFORE the first compile wraps it) and calls
    `_launcher(kernel, grid, *args, num_warps=, num_stages=)`, dropping the return value. Triton is
    stubbed (no Linux wheel on a Mac); skipped when real Triton is installed (the PTX CI test covers it)."""
    helion = pytest.importorskip("helion")
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("real Triton present: the stub would shadow it; tests/test_fp4mma_ptx.py covers this")
    from helion.runtime.ref_mode import RefMode
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm

    class Jit:  # what @triton.jit returns: an object with .fn
        def __init__(self, fn):
            self.fn = fn

    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **kw: Jit(fn) if fn is not None else Jit
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (n - 1).bit_length()
    tl = types.ModuleType("triton.language")
    tl.constexpr = type("constexpr", (), {"__init__": lambda self, v=None: setattr(self, "value", v)})
    triton.language = tl
    before = set(sys.modules)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", tl)
    try:
        p = fm.problem({"k": 256, "n": 64, "numtokens": 5}, 0, "cpu", fm.WORDS)
        args = fm.body_args(fm.WORDS, p)
        kern = helion.kernel(fm.BODIES[fm.WORDS], ref_mode=RefMode.OFF, static_shapes=False,
                             ignore_warnings=fm._ignore())
        code = kern.bind(args).to_triton_code(helion.Config(split_k=2, block_sizes=[16, 32, 32]))
        assert "from helion.runtime import default_launcher as _default_launcher" in code
        probe.install_recorder()  # BEFORE the module's first import, like prepare() does
        path = tmp_path / "gen_words.py"
        path.write_text(code)
        spec = importlib.util.spec_from_file_location("gen_words_k5", path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "gen_words_k5", mod)
        spec.loader.exec_module(mod)
        assert mod.nvfp4_gemm_q_fp4mma.__kwdefaults__["_launcher"].__name__ == "recording_launcher"
        mod.nvfp4_gemm_q_fp4mma(*args)
        r, = probe.records("_helion_nvfp4_gemm_q_fp4mma")
        assert r.name == "_helion_nvfp4_gemm_q_fp4mma" and r.num_warps == 4
        probe.assert_native([fm.TRITON_NAMES[fm.WORDS]], log=lambda s: None)
    finally:
        for name in set(sys.modules) - before:
            if name.startswith(("helion.runtime.triton", "gen_words_k5")):
                sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# capture / replay machinery
# ---------------------------------------------------------------------------
def test_capture_check_passes_a_pure_function_and_bites_carried_state():
    xs = [torch.full((4,), float(i), dtype=torch.float32).bfloat16() for i in range(3)]
    copy_in = lambda st, x: st.copy_(x)
    pure = lambda st: st * 2
    assert probe.capture_check(pure, xs, copy_in, probe.EagerRuntime, 6) == 6
    state = {"calls": 0}

    def carried(st):  # a counter that survives replays: a "ticket" that is not back at zero
        state["calls"] += 1
        return st + state["calls"]
    with pytest.raises(probe.Fp4mmaError, match="CAPTURE/REPLAY FAIL"):
        probe.capture_check(carried, xs, copy_in, probe.EagerRuntime, 4, what="ticket")


def test_guarded_output_detects_a_stray_store_on_either_side():
    arena, view = probe.guarded(3, 8, "cpu")
    assert probe.guards_intact(arena) and view.shape == (3, 8) and view.dtype == torch.bfloat16
    view.fill_(1.0)
    assert probe.guards_intact(arena)
    arena[probe.GUARD - 1] = 0
    assert not probe.guards_intact(arena)
    arena2, _ = probe.guarded(3, 8, "cpu")
    arena2[-1] = 0
    assert not probe.guards_intact(arena2)


def test_boot_gate_argv_parses_with_the_gates_own_parser():
    argv, env = probe.BOOT_GATES["fp4mma_capture_replay"]
    assert argv[:2] == ["-m", "suffix_hybrid.kernels.fp4mma_probe"]
    ns = probe.parser().parse_args(argv[2:])  # a flag typo cannot ship
    assert ns.cmd == "capture_replay"
    assert env == {"SUFFIX_HELION_NVFP4_GEMM_FP4MMA": "1", "SUFFIX_HELION_NVFP4_MOE_FP4MMA": "1"}
    assert probe.main(["capture_replay"]) == 2  # no CUDA here -> NOT RUN, not a crash
    # the bytes-mode gate: the same run, the scale-read mode in the child's env (sitecustomize strips SUFFIX_*)
    argv_b, env_b = probe.BOOT_GATES["fp4mma_capture_replay_bytes"]
    assert argv_b == argv and env_b == {**env, "SUFFIX_FP4MMA_MOE_SCALE_READ": "bytes"}


def test_probe_cli_ptx_helper(tmp_path, capsys):
    path = tmp_path / "k.ptx"
    path.write_text(NATIVE_PTX)
    assert probe.main(["ptx", str(path)]) == 0
    assert '"native_mma": 1' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# P0 script + bake-off harness (pure parts)
# ---------------------------------------------------------------------------
def test_p0_script_selftest_runs_without_triton():
    r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "fp4mma_p0_probe.py"), "--selftest"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0 and "selftest OK" in r.stdout, r.stderr[-800:]


def load_p0():
    spec = importlib.util.spec_from_file_location("fp4mma_p0_probe", os.path.join(ROOT, "scripts", "fp4mma_p0_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_p0_mirror_kernel_source_is_wellformed_and_covers_every_pattern():
    import ast
    p0 = load_p0()
    tree = ast.parse(p0.MIRROR_SRC)
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert funcs == {"mirror", "mirror_cvt"}
    # every variant maps to a branch the kernel has; the primary set is the PR kernel and the Helion words body
    assert set(p0.VARIANTS) == {"pr", "pr_perm", "bytes", "words", "linwords"} and p0.PRIMARY == ("pr", "words")
    assert {v[0] for v in p0.VARIANTS.values()} == {0, 1, 2, 3}
    for needle in ("tl.dot_scaled(a, sa, \"e2m1\", b, sb, \"e2m1\"", "lhs_k_pack=True", "rhs_k_pack=True",
                   "tl.float8e4nv, bitcast=True", "tl.permute"):
        assert needle in p0.MIRROR_SRC
    assert len(p0.configs(True)) == 12 and len(p0.configs(False)) == 576
    assert all(c[0] in (16, 32, 64, 128) and c[2] in (64, 128, 256) for c in p0.configs(False))


def test_p0_mirror_cvt_asm_is_helions_asm():
    """The cvt mirror embeds a copy of Helion's inline asm; pin it against the installed Helion."""
    pytest.importorskip("helion")
    from helion._compiler.triton.quantized_ops import _FP4X16_TO_F16_ASM
    p0 = load_p0()
    asm = p0.MIRROR_SRC.split('FP4X16_TO_F16_ASM = tl.constexpr("""')[1].split('"""')[0]
    assert asm.strip() == _FP4X16_TO_F16_ASM.strip()


def test_p0_helion_mode_generates_every_real_body_for_gemm_and_moe():
    """`--helion` compiles Helion's generated Triton for sm_120a (Linux). Its CPU half runs here: the
    front end must lower all nine real bodies (4 dense GEMM + 5 MoE) at an explicit config to parseable
    Triton source with the dot_scaled calls K5 will look for, and flag exactly fc1 / fc2 / the e2m1 GEMM
    bodies as native-expected."""
    import ast
    import types
    pytest.importorskip("helion")
    p0 = load_p0()
    ns = types.SimpleNamespace(helion_families="gemm,moe", n=64, k=256, moe_e=4, moe_h=256, moe_i=128)
    rows = {label: (code, want) for label, _kind, code, _args, want in p0.helion_bodies(ns)}
    moe = {"moe.dsv41_fp4mma_route_quant", "moe.dsv41_fp4mma_quant", "moe.dsv41_fp4mma_fc1",
           "moe.dsv41_fp4mma_fc2", "moe.dsv41_fp4mma_combine"}
    assert set(rows) == {"gemm.nvfp4_gemm_q_fp4mma", "gemm.nvfp4_gemm_q_fp4mma_b", "gemm.nvfp4_gemm_q_fp4mma_lin",
                         "gemm.nvfp4_gemm_q_fp4cvt"} | moe | {"moe.dsv41_fp4mma_fc1[bytes]",
                                                              "moe.dsv41_fp4mma_fc2[bytes]"}
    dots = lambda code: sum(1 for ln in code.splitlines() if "tl.dot_scaled(" in ln and not ln.strip().startswith("#"))
    for label, (code, want) in rows.items():
        ast.parse(code)
        n_dot = {"moe.dsv41_fp4mma_fc1": 2, "moe.dsv41_fp4mma_fc2": 1, "moe.dsv41_fp4mma_fc1[bytes]": 2,
                 "moe.dsv41_fp4mma_fc2[bytes]": 1}.get(label, 1 if want else 0)
        assert dots(code) == n_dot, (label, dots(code))
        assert want == (label in ("gemm.nvfp4_gemm_q_fp4mma", "gemm.nvfp4_gemm_q_fp4mma_b",
                                  "gemm.nvfp4_gemm_q_fp4mma_lin", "moe.dsv41_fp4mma_fc1", "moe.dsv41_fp4mma_fc2",
                                  "moe.dsv41_fp4mma_fc1[bytes]", "moe.dsv41_fp4mma_fc2[bytes]"))
        # the two scale-read modes really are two different programs (join/bitcast unpack vs plain e4m3 loads)
        if label.endswith("[bytes]"):
            assert "tl.join" not in code and "bitcast=True" not in code, label
        elif label in ("moe.dsv41_fp4mma_fc1", "moe.dsv41_fp4mma_fc2"):
            assert "tl.join" in code and "bitcast=True" in code, label
    ns.helion_families = "gemm"
    assert len(p0.helion_bodies(ns)) == 4
    from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fe
    assert fe.SCALE_READ == "words"  # helion_bodies restores the mode it flipped


def test_p0_summary_legal_tiles_and_bounds():
    p0 = load_p0()
    info = probe.analyze_ptx(NATIVE_PTX)
    ok = lambda bm, bn, bk, **kw: p0.result_row("words", (bm, bn, bk, 4, 3), info, {"regs": 80, "smem": 1, "spill_stores": 0,
                                                                                  "spill_loads": 0, **kw}, 30000, 1.0, [])
    rows = [ok(16, 64, 256), ok(16, 64, 128), ok(64, 128, 256), ok(128, 128, 256, spill_stores=16),
            dict(ok(32, 64, 64), native=False, problems=["decomposed"]),
            {"variant": "words", "bm": 16, "bn": 256, "bk": 256, "warps": 4, "stages": 3, "error": "boom",
             "native": False, "problems": []}]
    s = p0.summarize(rows)["words"]
    assert (s["legal"], s["errors"], s["total"]) == (3, 1, 6)
    assert s["suggested_bounds"] == {"BM": [16, 64], "BN": [64, 128], "BK_bytes": [64, 128]}
    assert s["decomposed"] == [(32, 64, 64, 4, 3)]
    big = ok(16, 64, 256, smem=1)
    big["problems"] = ["smem 200000 > 101376"]
    assert not p0.legal(big)


def test_bench_selftest_and_screen_definitions():
    from suffix_hybrid.kernels import fp4mma_bench as fb
    assert fb.selftest() == 0
    from suffix_hybrid.kernels import nvfp4_gemm as ng
    cases = fb.screen_cases("21")
    assert len(cases) == 21 and {c[3] for c in cases} == {1, 16, 64}
    assert {(c[1], c[2]) for c in cases} == set(ng.SHAPES.values()) | {fb.QWEN_LM_HEAD}
    assert {c[1:3] for c in fb.screen_cases("dsv41")} == {fb.V41_LM_HEAD}
    assert len(fb.screen_cases("all")) == 21 + 8 + 18
    with pytest.raises(ValueError):
        fb.screen_cases("nope")


def test_assert_contains_for_the_cvt_control_arm(fake_launcher):
    import helion.runtime as hrt
    probe.install_recorder()
    cvt_ptx = NATIVE_PTX.replace(probe.NATIVE_MMA, "cvt.rn.f16x2.e2m1x2 %r1, %rs1;\n    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
    fake_launcher["ck"] = FakeCK(cvt_ptx, "c1")
    hrt.default_launcher(FakeJit("_helion_nvfp4_gemm_q_fp4cvt"), (1,))
    got = probe.assert_contains(["_helion_nvfp4_gemm_q_fp4cvt"], ["cvt.rn.f16x2.e2m1x2"], "hw decode",
                                log=lambda s: None)
    assert len(got) == 1
    with pytest.raises(probe.Fp4mmaError, match="missing"):
        probe.assert_contains(["_helion_nvfp4_gemm_q_fp4cvt"], ["cvt.rn.f16x2.e4m3x2"], "hw decode")
    # the cvt arm is NOT native: asking assert_native about it fails (so verify_served never does)
    with pytest.raises(probe.Fp4mmaError, match="decomposed"):
        probe.assert_native(["_helion_nvfp4_gemm_q_fp4cvt"])
