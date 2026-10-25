"""Ordered NVFP4 fallback: replay pinned upstream call sites without a GPU."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from suffix_hybrid import rocm_patches as rp

PATCH = rp.PATCHES["SUFFIX_NVFP4_MOE_ORDERED_FINALIZE"]
FIXTURES = Path(__file__).resolve().parents[1] / "sm120/tests/fixtures"


@pytest.mark.parametrize("version", ["0.30.0", "0.32.0"])
def test_upstream_call_and_quantization_scope(version):
    src = (FIXTURES / f"vllm_{version}/flashinfer_cutlass_moe.py").read_text()
    expected = {
        "0.30.0": "52887cec5fe628e0a4e7793deb546b40f6c29f7362b3b436c10ca36934ab9826",
        "0.32.0": "82e1784f277df2d4e079b61c3af68fa11375e0f71f6cd9a7a95111c885e75963",
    }
    assert hashlib.sha256(src.encode()).hexdigest() == expected[version]
    changed = rp.patch_source(PATCH, src)
    tree = ast.parse(changed)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "flashinfer_cutlass_fused_moe"
    ]
    assert len(calls) == 1
    kw = next(k.value for k in calls[0].keywords if k.arg == "use_fused_finalize")
    expr = compile(ast.Expression(kw), "<finalize>", "eval")
    for dtype, expected in [
        ("nvfp4", False),
        ("mxfp4", True),
        ("fp8", True),
        (None, True),
    ]:
        assert eval(expr, {"self": SimpleNamespace(quant_dtype=dtype)}) is expected
    with pytest.raises(RuntimeError, match="refusing to serve unpatched"):
        rp.patch_source(PATCH, changed)
    with pytest.raises(RuntimeError, match="refusing to serve unpatched"):
        rp.patch_source(PATCH, src.replace(PATCH.old, "# changed upstream\n"))


def test_gate_defaults_off_and_arms_only_explicitly(monkeypatch):
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    assert not rp.enabled()
    monkeypatch.setenv("SUFFIX_NVFP4_MOE_ORDERED_FINALIZE", "1")
    assert rp.enabled() == {PATCH.target: [PATCH]}
