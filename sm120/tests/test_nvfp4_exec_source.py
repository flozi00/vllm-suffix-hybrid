# SPDX-License-Identifier: Apache-2.0
"""The patched backend must stay source-introspectable: Triton @jit reads
kernel source via inspect.getsource (gemma-spec-dev nvfp4 boot 2026-09-24:
"@jit functions should be defined in a Python file")."""

import inspect
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvfp4_kv_patch import exec_patched_source  # noqa: E402


def test_rewritten_functions_keep_their_source(tmp_path):
    orig = tmp_path / "backend.py"
    orig.write_text("def f():\n    return 1\n")
    module = types.ModuleType("backend")
    module.__file__ = str(orig)
    # The rewrite is longer than the file: g sits past the original's last
    # line, exactly where compile-under-the-original-name broke getsource.
    new_src = ("# injected helper\n" * 50
               + "def f():\n    return 2\n\n"
               + "def g(x):\n    return x + 1\n")
    exec_patched_source(module, new_src, orig)
    assert module.f() == 2
    assert inspect.getsource(module.g) == "def g(x):\n    return x + 1\n"
    assert inspect.getsource(module.f) == "def f():\n    return 2\n"
