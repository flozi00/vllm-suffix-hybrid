# SPDX-License-Identifier: Apache-2.0
"""Opt-in pinned runner hook; an explicit enable request must fail closed."""
import os

if os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() == "1":
    try:
        # V2 model runner first (the platform default on recent vLLM): the
        # speculator-path hook. Returns False only when the V2 runner module
        # does not exist in this vLLM, in which case the V1 class hook below
        # is the right target.
        from suffix_hybrid.wrap_v2 import install_v2
        if not install_v2():
            from suffix_hybrid.wrap import install
            install()
    except Exception as exc:
        # site.py swallows Exception and continues unpatched. SystemExit is a
        # BaseException, so an incompatible enabled worker cannot silently run.
        raise SystemExit(f"suffix hybrid installation failed: {exc}") from exc
