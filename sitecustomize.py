# SPDX-License-Identifier: Apache-2.0
"""sitecustomize entry points: dev_args splice, kernel registration, wrap.

site.py runs this at interpreter start of EVERY process that has /plugins on
PYTHONPATH. Each section is independently gated:
  SUFFIX_HYBRID_DEV_ARGS=1  -> /plugins/dev_args.json argv splice (serve only)
  SUFFIX_KERNELS=1          -> register Triton IR-op providers (inert until
                               --kernel-config ir_op_priority names them)
  SUFFIX_HYBRID_WRAP=1      -> speculator wrap (fail closed on drift)
Prod sets none of them, so all three are inert there. The kernels gate lives
OUTSIDE the wrap's fail-closed try: a kernel registration failure must degrade
to vllm_c/native with a logged refusal, never kill an otherwise healthy pool.
"""
import os

# Kernel provider registration (independent gate). Inert until
# --kernel-config ir_op_priority names the provider; a registration failure
# is a logged refusal, never fatal (serving never depended on our ops).
if os.environ.get("SUFFIX_KERNELS", "").strip() == "1":
    try:
        from suffix_hybrid.kernels.install import install_kernels
        install_kernels()
    except BaseException as exc:  # noqa: BLE001 - logged refusal by design
        # NOT SystemExit (contrast with the wrap below): the serving path
        # never depended on our kernels, so an import-time explosion here
        # must degrade to vllm_c/native, not kill a healthy pool.
        import sys
        print(f"suffix kernels installation FAILED (degrading to native "
              f"backends): {exc}", file=sys.stderr, flush=True)

# Dev-mode argv splice (independent gate): iterate on serving flags without a
# pod roll. Splices /plugins/dev_args.json into sys.argv before vllm's parser
# runs; applies only to the `vllm serve` entrypoint (dev_args.py), never to
# worker children. Prod never sets the gate, so this is inert there.
if os.environ.get("SUFFIX_HYBRID_DEV_ARGS", "").strip() == "1":
    from suffix_hybrid.dev_args import apply_dev_args
    apply_dev_args()

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

# sm120 deep_gemm shim status. The bundle always ships /plugins/deep_gemm/
# (it shadows site-packages' deep_gemm via PYTHONPATH=/plugins and
# self-delegates to the vendored DeepGEMM unless SUFFIX_SM120=1 on an SM120
# GPU), so nothing is imported here — the shim resolves itself lazily in the
# one process that actually does `import deep_gemm`. One log line per process
# so operators can tell active vs inert from the pod log without torch ever
# loading at sitecustomize time.
if os.path.isdir("/plugins/deep_gemm"):
    import sys
    if os.environ.get("SUFFIX_SM120", "").strip() == "1":
        print("suffix sm120: deep_gemm shim active (/plugins/deep_gemm on "
              "sys.path ahead of site-packages)", file=sys.stderr, flush=True)
    else:
        print("suffix sm120: deep_gemm shim present but inert "
              "(SUFFIX_SM120 unset; delegates to vendored DeepGEMM)",
              file=sys.stderr, flush=True)

