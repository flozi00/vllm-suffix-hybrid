"""SUFFIX_BOOT_GATES: allowlisted, run once per pod, feature gates stripped from children."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _boot(extra_env, tmp_path, package="nvfp4_ds_mla_patch"):
    # A fake `<package>.oracle` that records the env it was run with.
    pkg = tmp_path / package
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "oracle.py").write_text(
        "import json, os, sys\n"
        f"open({str(tmp_path / 'seen.json')!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], "
        "'env': {k: v for k, v in os.environ.items() if k.startswith('SUFFIX_')}}))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    env.update(extra_env, PYTHONPATH=f"{REPO}{os.pathsep}{tmp_path}")
    return subprocess.run([sys.executable, "-c", "import sitecustomize"], env=env,
                          capture_output=True, text=True, cwd=tmp_path)


def test_unknown_gate_is_skipped(tmp_path):
    r = _boot({"SUFFIX_BOOT_GATES": "sh -c id"}, tmp_path)
    assert r.returncode == 0 and "unknown gate, skipped" in r.stderr
    assert not (tmp_path / "seen.json").exists()


def test_gate_runs_once_with_feature_gates_stripped(tmp_path):
    import json
    r = _boot({"SUFFIX_BOOT_GATES": "nvfp4_dsmla_bench", "SUFFIX_NVFP4_MOE": "1"}, tmp_path)
    assert "[suffix boot-gate] nvfp4_dsmla_bench: exit 0" in r.stderr, r.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["argv"] == ["--bench", "--json"]
    assert "SUFFIX_NVFP4_MOE" not in seen["env"] and seen["env"].get("SUFFIX_BOOT_GATES_DONE") == "1"


def test_done_marker_suppresses_rerun(tmp_path):
    r = _boot({"SUFFIX_BOOT_GATES": "nvfp4_dsmla_bench", "SUFFIX_BOOT_GATES_DONE": "1"}, tmp_path)
    assert "boot-gate" not in r.stderr and not (tmp_path / "seen.json").exists()


def test_glm_stack_tp2_gate_is_the_tp1_gate_plus_tp2():
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    argv, env = ns["_BOOT_GATES"]["glm_stack_tp2_oracle"]
    assert argv == ns["_BOOT_GATES"]["glm_stack_oracle"][0] + ["--tp", "2"]
    assert env == {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1"}


def test_glm_stack_multi_oracle_gate(tmp_path):
    import json
    r = _boot({"SUFFIX_BOOT_GATES": "glm_stack_multi_oracle",
               "SUFFIX_SM120_HISPARSE_PREFETCH": "1"}, tmp_path, "hisparse_mtp_patch")
    assert "[suffix boot-gate] glm_stack_multi_oracle: exit 0" in r.stderr, r.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["argv"][0] == "--multi" and "nvfp4_ds_mla" in seen["argv"]
    # The oracle sets the patch gates per child itself; nothing else leaks in.
    assert {k: v for k, v in seen["env"].items() if not k.endswith("_DONE")} == {
        "SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1"}
