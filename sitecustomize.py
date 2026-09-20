# SPDX-License-Identifier: Apache-2.0
"""Auto-imported by CPython when this directory is on PYTHONPATH.

Installs the suffix_hybrid wrap-mode meta-path hook. A no-op unless
SUFFIX_HYBRID_WRAP=1, and can never raise: a broken sitecustomize would
break the entire pod.
"""

try:
    if __package__ in (None, ""):
        import sys as _sys
        import os as _os

        _dir = _os.path.dirname(_os.path.abspath(__file__))
        if _dir not in _sys.path:
            _sys.path.insert(0, _dir)

    from suffix_hybrid import wrap as _wrap

    _wrap.install()
except Exception:  # never propagate
    pass