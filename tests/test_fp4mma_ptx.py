# SPDX-License-Identifier: Apache-2.0
"""Linux CI only (skipped without Triton): the P0 probe as a test. Compiles the raw-Triton mirror of the
exact `tl.dot_scaled` e2m1 x e2m1 / e4m3-scale call (plus the scale-unpack patterns our Helion bodies
emit) for GPUTarget("cuda", 120, 32) with whatever Triton wheel the runner has (the native.yml test step
installs 3.8.0), no GPU needed, and asserts the PTX is the native `kind::mxf4nvf4` instruction for
`.target sm_120a`, spill-free, inside 99 KB of shared memory.

A failure here is the finding "Triton silently decomposes this pattern on this version": fix the pattern
(bodies in nvfp4_gemm_fp4mma.py) or ship only the variants that pass. Not run by its author (macOS).
"""
import importlib.util
import os
import sys

import pytest

triton = pytest.importorskip("triton", reason="Triton wheels are Linux-only; this is the Linux CI gate")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if sys.platform != "linux":
    pytest.skip("Triton PTX gate runs on Linux", allow_module_level=True)

from suffix_hybrid.kernels import fp4mma_probe as probe  # noqa: E402


def load_p0():
    spec = importlib.util.spec_from_file_location("fp4mma_p0_probe", os.path.join(ROOT, "scripts", "fp4mma_p0_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p0():
    mod = load_p0()
    if (probe.parse_version(triton.__version__) or (0, 0, 0)) < probe.MIN_TRITON:
        pytest.skip(f"triton {triton.__version__} < {probe.MIN_TRITON}")
    try:
        mod.ptxas_path(120)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no ptxas-blackwell in this Triton: {exc}")
    return mod


CONFIGS = [(16, 64, 256, 4, 4), (16, 32, 128, 4, 3), (64, 128, 256, 4, 4)]


@pytest.mark.parametrize("variant", ["pr", "pr_perm", "bytes", "words", "linwords"])
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "x".join(map(str, c)))
def test_mirror_lowers_to_the_native_fp4_mma(p0, tmp_path, variant, cfg):
    row = p0.compile_one({"variant": variant, "cfg": list(cfg), "arch": 120, "k": 5120, "n": 4096,
                          "tmpdir": str(tmp_path), "save_ptx": ""})
    assert not row.get("error"), row.get("error")
    assert row["native"], (f"{variant} {cfg} on triton {triton.__version__}: not the native instruction "
                           f"(other mma: {row['other_mma']}); problems {row['problems']}")
    assert row["target"] == "sm_120a" and row["n_native_mma"] >= 1
    assert not row["problems"], row["problems"]
    assert (row["spill_stores"] or 0) == 0 and (row["spill_loads"] or 0) == 0, row
    assert 0 < (row["smem_triton"] or 1) <= probe.SM120_SMEM


def test_cvt_mirror_assembles_for_sm120a(p0, tmp_path):
    """cvt.rn.f16x2.e2m1x2 exists on sm_120a (PTX ISA 8.6+): ptxas must accept the inline asm."""
    row = p0.compile_one({"variant": "cvt", "cfg": [16, 64, 128, 4, 3], "arch": 120, "k": 5120, "n": 4096,
                          "tmpdir": str(tmp_path), "save_ptx": ""})
    assert not row.get("error"), row.get("error")
    assert not row["problems"], row["problems"]


def test_real_helion_bodies_compile_for_sm120a_with_the_native_mma(p0):
    """The P0 `--helion` mode as a test: Helion's REAL generated kernels (4 dense GEMM bodies, the 3 MoE bodies,
    fc1 / fc2 in both scale-read modes) go through JITFunction.run with a duck-typed driver and are compiled for
    GPUTarget("cuda", 120, 32): every dot_scaled body must carry the native instruction. Register spills are
    printed (-s), not failed: they are a tuning fact of the probe config, not a lowering bug."""
    pytest.importorskip("helion")
    import types
    ns = types.SimpleNamespace(arch=120, n=4096, k=5120, helion_families="gemm,moe", moe_e=4, moe_h=5120,
                               moe_i=2304)
    rows = p0.run_helion(ns)
    assert len(rows) == 9, [r["variant"] for r in rows]
    for r in rows:
        assert not r.get("error"), f"{r['variant']}: {r['error']}"
        problems = [x for x in r["problems"] if not x.startswith("local memory")]
        assert not problems, f"{r['variant']} on triton {triton.__version__}: {problems}"
        print(f"{r['variant']}: regs {r['regs']} spills {r['spill_stores']}/{r['spill_loads']} smem {r['smem_triton']} "
              f"native_mma x{r['n_native_mma']}")
