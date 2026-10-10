"""SUFFIX_ROCM_PRESET: sitecustomize setdefaults a validated gate set at interpreter start."""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SUFFIX_", "VLLM_ROCM_"))}
    env.update(PYTHONPATH=str(ROOT), **env_extra)
    code = ("import os; print(os.environ.get('SUFFIX_ROCM_GDN_DEFER_MFMA'), "
            "os.environ.get('SUFFIX_ROCM_HC_DOWN_MAX_M'), os.environ.get('VLLM_ROCM_USE_AITER'))")
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_preset_sets_gates_and_explicit_env_wins():
    r = _run({"SUFFIX_ROCM_PRESET": "mi350p-qwen-flash", "SUFFIX_ROCM_HC_DOWN_MAX_M": "64"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["1", "64", "1"]


def test_unknown_preset_refuses_to_start():
    r = _run({"SUFFIX_ROCM_PRESET": "nope"})
    assert r.returncode != 0 and "unknown" in r.stderr


def test_no_preset_is_inert():
    r = _run({})
    assert r.returncode == 0 and r.stdout.split() == ["None", "None", "None"]
