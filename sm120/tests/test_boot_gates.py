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
    env.update(extra_env, PYTHONPATH=f"{REPO}{os.pathsep}{tmp_path}",
               SUFFIX_BOOT_GATES_MARKER=str(tmp_path / "gates.claimed"))
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


def test_marker_file_suppresses_rerun_in_a_second_top_level_process(tmp_path):
    # `vllm serve` starts >1 top-level Python; the env marker can't reach a sibling.
    _boot({"SUFFIX_BOOT_GATES": "nvfp4_dsmla_bench"}, tmp_path)
    (tmp_path / "seen.json").unlink()
    r = _boot({"SUFFIX_BOOT_GATES": "nvfp4_dsmla_bench"}, tmp_path)
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


def test_nvfp4_moe_gates_run_the_kernel_module_cli():
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    for mode in ("oracle", "bench", "sweep"):
        argv, env = ns["_BOOT_GATES"][f"nvfp4_moe_{mode}"]
        assert argv == ["-m", "suffix_hybrid.kernels.nvfp4_moe", mode] and env == {}
    # the module's CLI dispatches exactly these modes (no GPU needed to check)
    mod = (REPO / "suffix_hybrid" / "kernels" / "nvfp4_moe.py").read_text()
    assert 'if mode in ("oracle", "both")' in mod and 'if mode in ("bench", "both")' in mod
    assert 'if mode in ("sweep",)' in mod


def test_nvfp4_moe_gate_runs_once_with_feature_gates_stripped(tmp_path):
    import json
    # Shadow the real module with a recorder (same dotted path, earlier on PYTHONPATH).
    kdir = tmp_path / "suffix_hybrid" / "kernels"
    kdir.mkdir(parents=True)
    (tmp_path / "suffix_hybrid" / "__init__.py").write_text("")
    (kdir / "__init__.py").write_text("")
    (kdir / "nvfp4_moe.py").write_text(
        "import json, os, sys\n"
        f"open({str(tmp_path / 'seen.json')!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], "
        "'env': {k: v for k, v in os.environ.items() if k.startswith('SUFFIX_')}}))\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    env.update({"SUFFIX_BOOT_GATES": "nvfp4_moe_oracle", "SUFFIX_NVFP4_MOE": "1"},
               PYTHONPATH=f"{tmp_path}{os.pathsep}{REPO}",
               SUFFIX_BOOT_GATES_MARKER=str(tmp_path / "gates.claimed"))
    r = subprocess.run([sys.executable, "-c", "import sitecustomize"], env=env,
                       capture_output=True, text=True, cwd=tmp_path)
    assert "[suffix boot-gate] nvfp4_moe_oracle: exit 0" in r.stderr, r.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["argv"] == ["oracle"] and "SUFFIX_NVFP4_MOE" not in seen["env"]


def test_nvfp4_kv_gates(tmp_path):
    import json
    r = _boot({"SUFFIX_BOOT_GATES": "nvfp4_kv_oracle_own,nvfp4_kv_bench",
               "SUFFIX_SM120_NVP4KV_OWN_ATTN": "1"}, tmp_path, "nvfp4_kv_patch")
    assert "[suffix boot-gate] nvfp4_kv_oracle_own: exit 0" in r.stderr, r.stderr
    assert "[suffix boot-gate] nvfp4_kv_bench: exit 0" in r.stderr, r.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())  # last gate run
    assert seen["argv"][:2] == ["--bench", "--json"] and "SUFFIX_SM120_NVP4KV_OWN_ATTN" not in seen["env"]
    # argv mirrors the kernel lab's K2 preset (--page 64 --max-gb 8 fixed)
    assert seen["argv"][-4:] == ["--page", "64", "--max-gb", "8"]


def test_nvfp4_lmhead_gates_are_allowlisted():
    src = (REPO / "sitecustomize.py").read_text()
    ns = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    for mode in ("oracle", "bench"):
        argv, env = ns["_BOOT_GATES"][f"nvfp4_lmhead_{mode}"]
        assert argv == ["-m", "suffix_hybrid.kernels.nvfp4_lm_head", mode] and env == {}


def test_hybrid_wrap_installs_after_the_sm120_patch_hooks():
    src = (REPO / "sitecustomize.py").read_text()
    wrap = src.index('if os.environ.get("SUFFIX_HYBRID_WRAP"')
    for hook in ("SUFFIX_SM120_NVP4KV\"", "SUFFIX_SM120_HISPARSE_MTP\"", "SUFFIX_SM120_NVP4DSMLA\""):
        i = src.find(f'os.environ.get("{hook[:-1]}"')
        assert i != -1 and i < wrap, hook


def test_nostage_gate_is_mixed_plus_its_env():
    src = (Path(__file__).resolve().parents[2] / "sitecustomize.py").read_text()
    ns: dict = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    argv, env = ns["_BOOT_GATES"]["glm_stack_nostage_oracle"]
    m_argv, m_env = ns["_BOOT_GATES"]["glm_stack_mixed_oracle"]
    assert argv == m_argv
    assert env == dict(m_env, SUFFIX_SM120_HISPARSE_NO_MIRROR_STAGING="1")


def test_fp8_dense_oracle_gate(tmp_path):
    import json
    r = _boot({"SUFFIX_BOOT_GATES": "fp8_dense_oracle", "SUFFIX_FP8_DENSE_LAYERS": "*"},
              tmp_path, "fp8_dense_patch")
    assert "[suffix boot-gate] fp8_dense_oracle: exit 0" in r.stderr, r.stderr
    seen = json.loads((tmp_path / "seen.json").read_text())
    assert seen["argv"] == [] and "SUFFIX_FP8_DENSE_LAYERS" not in seen["env"]


def test_fp8_dense_hook_arms_before_the_hybrid_wrap():
    src = (REPO / "sitecustomize.py").read_text()
    i = src.find('os.environ.get("SUFFIX_FP8_DENSE"')
    assert i != -1 and i < src.index('if os.environ.get("SUFFIX_HYBRID_WRAP"')


def test_nvfp4_dense_oracle_gate_and_hook():
    src = (REPO / "sitecustomize.py").read_text()
    ns: dict = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], ns)
    argv, env = ns["_BOOT_GATES"]["nvfp4_dense_oracle"]
    assert argv == ["-m", "fp8_dense_patch.nvfp4_oracle"] and env == {}
    assert (REPO / "sm120/fp8_dense_patch/nvfp4_oracle.py").is_file()
    i = src.find('os.environ.get("SUFFIX_NVFP4_DENSE"')
    assert i != -1 and i < src.index('if os.environ.get("SUFFIX_HYBRID_WRAP"')
