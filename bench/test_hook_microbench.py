# SPDX-License-Identifier: Apache-2.0
"""Fast contract tests for bench/hook_microbench.py.

These pin the harness plumbing against the BASELINE snapshot module (the
c20bec73 wrap_v2.py materialized under the system temp dir), never the live
suffix_hybrid/wrap_v2.py — another agent is rewriting that file. Steps are
kept tiny (~20); this proves the fake runner/TP stub and arms produce
positive numbers, it is not a measurement run.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import hook_microbench as hb

BASELINE_REV = "c20bec73"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _baseline_module():
    """Import the HEAD-pinned baseline wrap_v2 from the temp snapshot.

    The snapshot is created once per machine (bench task step 1); if the temp
    dir was reaped, re-materialize it straight from git without touching the
    working tree. The live wrap_v2.py is never imported here.
    """
    pkg = Path(tempfile.gettempdir()) / "sh_baseline_pkg"
    target = pkg / "wrap_v2.py"
    if not target.exists():
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").touch()
        src = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show",
             f"{BASELINE_REV}:suffix_hybrid/wrap_v2.py"],
            capture_output=True, text=True, check=True).stdout
        target.write_text(src)
    if str(pkg.parent) not in sys.path:
        sys.path.insert(0, str(pkg.parent))
    import importlib
    module = importlib.import_module("sh_baseline_pkg.wrap_v2")
    # Guard: we must be loading the snapshot, not the live tree's module.
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == target.resolve()
    return module


@pytest.fixture(scope="module")
def baseline():
    return _baseline_module()


def test_baseline_snapshot_warm_arm_positive(baseline):
    res = hb.run_arm(baseline, rows=4, context=1024, steps=20, arm="warm")
    assert res["mean_us"] > 0
    assert res["median_us"] > 0
    assert res["p99_us"] >= res["median_us"]
    # The corpus feed + finalize path must have engaged the real mixer.
    assert res["mixer_calls"] > 0
    assert res["cache_tokens"] > 0


def test_baseline_snapshot_cold_arm_pays_full_cost(baseline):
    res = hb.run_arm(baseline, rows=4, context=1024, steps=20, arm="cold")
    assert res["mean_us"] > 0
    assert res["p99_us"] > 0
    # Headline baseline property: an unfed cache still pays every wire-up
    # step (mixer IS called) while holding zero corpus. The rewrite's cold
    # path must instead skip the work entirely.
    assert res["mixer_calls"] > 0
    assert res["cache_tokens"] == 0
