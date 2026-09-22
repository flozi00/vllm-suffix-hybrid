# SPDX-License-Identifier: Apache-2.0
"""Opt-in pinned runner hook; an explicit enable request must fail closed."""
import os

if os.environ.get("SUFFIX_HYBRID_WRAP", "").strip() == "1":
    try:
        # Dev pools (SUFFIX_HYBRID_DEV_ARGS=1) may carry a /plugins/
        # dev_args.json delta that splices serving flags into sys.argv
        # before vllm's parser runs — flag iteration without a pod roll.
        # Prod never sets the gate, so this is inert there.
        if os.environ.get("SUFFIX_HYBRID_DEV_ARGS", "").strip() == "1":
            from suffix_hybrid.dev_args import apply_dev_args
            apply_dev_args()
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
