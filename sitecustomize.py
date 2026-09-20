# SPDX-License-Identifier: Apache-2.0
"""Opt-in pinned runner hook; an explicit enable request must fail closed."""
import os

if os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() == "1":
    try:
        from suffix_hybrid.wrap import install
        install()
    except Exception as exc:
        # site.py swallows Exception and continues unpatched. SystemExit is a
        # BaseException, so an incompatible enabled worker cannot silently run.
        raise SystemExit(f"suffix hybrid installation failed: {exc}") from exc
