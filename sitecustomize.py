# SPDX-License-Identifier: Apache-2.0
"""sitecustomize entry points: dev_args splice, kernel registration, wrap.

site.py runs this at interpreter start of EVERY process that has /plugins on
PYTHONPATH. Each section is independently gated:
  SUFFIX_HYBRID_DEV_ARGS=1  -> /plugins/dev_args.json argv splice (serve only)
  SUFFIX_KERNELS=1          -> register Triton IR-op providers (inert until
                               --kernel-config ir_op_priority names them)
  SUFFIX_HYBRID_WRAP=1      -> speculator wrap (fail closed on drift)
  SUFFIX_SM120_NVP4KV=1     -> arm the deferred SM120 NVFP4-KV backend patch
                               (fail closed on SM120, inert elsewhere)
  SUFFIX_SM120_HISPARSE_MTP=1 -> arm the deferred SM120 HiSparse+MTP
                               builder patch (fail closed on SM120)
  SUFFIX_SM120_NVP4DSMLA=1  -> arm the deferred SM120 nvfp4_ds_mla sparse-MLA
                               KV patch (our kernels; fail closed on SM120)
  SUFFIX_SM120_NVP4KV_ORACLE=1 -> run the NVFP4-KV on-silicon oracle once per
                               pod before serving (fatal only with NVP4KV=1)
  SUFFIX_FICACHE=seed|dump|both -> FlashInfer autotune cache seed/harvest
                               (inert otherwise, degrades on failure)
  SUFFIX_PROFILE_STEPS=<N>:<skip>[:<per>] -> in-pod torch.profiler summary
                               ("[suffix-prof]" lines; fail-soft)
Prod sets none of them, so all five are inert there. The kernels gate lives
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
    import sys

    try:
        from suffix_hybrid.sched_sync import install_post_import_hook
        install_post_import_hook()
        main_gate = True
    except Exception as exc:
        print(f"sched_sync hook arm FAILED (async-scheduling force stays "
              f"-- WATCH ENGINE-CONFIG): {exc}", file=sys.stderr, flush=True)
        main_gate = False
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


# FlashInfer autotune cache seed/harvest (SM120 MoE tactic persistence).
# SUFFIX_FICACHE=seed|dump|both; inert otherwise. seed = write bundled
# autotune_configs.json before warmup if missing (config-hash keyed, so a
# mismatch can only no-op). dump = log the cache file as a marker line so the
# console log reader can harvest it for the next bundle seed. A failure here
# must never kill a healthy pool: degrade to stock vLLM cache behaviour.
# Loaded by file location (/plugins is on PYTHONPATH but `sm120` is a repo
# namespace, not a bundle package — the module ships flat at /plugins/ficache.py).
_ficache = os.environ.get("SUFFIX_FICACHE", "").strip().lower()
if _ficache in ("seed", "dump", "both"):
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "suffix_ficache", "/plugins/ficache.py")
        if _spec is None or _spec.loader is None:
            raise ImportError("/plugins/ficache.py not found in bundle")
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.install(_ficache)
    except BaseException as exc:  # noqa: BLE001 - degrade by design
        import sys
        print(f"suffix ficache installation FAILED (stock autotune cache "
              f"path stays in use): {exc}", file=sys.stderr, flush=True)

# sm120 NVFP4-KV patch (independent gate). vLLM 0.30 gates --kv-cache-dtype
# nvfp4 on trtllm-gen, which has no SM120 nvfp4 prefill cubins; this patch
# re-routes NVFP4 KV through the FlashInfer wrapper's fa2 backend on cc 12.x.
# The backend module is not importable at sitecustomize time, so we arm a
# stdlib-only sys.meta_path post-import hook; the real SM120 check, the
# FlashInfer feature probe, and the (fail-closed) source patch run the first
# time vllm.v1.attention.backends.flashinfer is imported — always before
# engine/backend construction. With the gate unset this is fully inert.
if os.environ.get("SUFFIX_SM120_NVP4KV", "").strip() == "1":
    try:
        from nvfp4_kv_patch import install_post_import_hook
        install_post_import_hook()
    except Exception as exc:
        # Fail closed ONLY on SM120: an enabled pool must never silently run
        # unpatched. Elsewhere (API servers, non-Blackwell pods that inherit
        # the env) the hook is irrelevant, so a broken import degrades to a
        # logged refusal, matching the kernels gate's posture.
        import sys
        _sm120 = False
        try:
            import torch
            _sm120 = (torch.cuda.is_available()
                      and torch.cuda.get_device_capability()[0] == 12)
        except Exception:
            pass
        if _sm120:
            raise SystemExit(
                f"suffix sm120 nvfp4-kv installation failed on SM120: {exc}"
            ) from exc
        print(f"suffix sm120 nvfp4-kv: hook install failed (not SM120, "
              f"ignoring): {exc}", file=sys.stderr, flush=True)

# sm120 HiSparse+MTP patch (independent gate, hisparse_mtp_patch/): lets the
# SM120 sparse-MLA builder classify spec-verify tokens as HiSparse decode
# (hot buffer) instead of prefill. Same deferred meta_path hook + fail-closed
# posture as NVP4KV; the SM120 check and the source rewrite run on first
# import of vllm's sparse_mla_attention module. Unset = fully inert.
if os.environ.get("SUFFIX_SM120_HISPARSE_MTP", "").strip() == "1":
    try:
        from hisparse_mtp_patch import install_post_import_hook as _hs_hook
        _hs_hook()
    except Exception as exc:
        import sys
        _sm120 = False
        try:
            import torch
            _sm120 = (torch.cuda.is_available()
                      and torch.cuda.get_device_capability()[0] == 12)
        except Exception:
            pass
        if _sm120:
            raise SystemExit(
                f"suffix sm120 hisparse-mtp installation failed on SM120: {exc}"
            ) from exc
        print(f"suffix sm120 hisparse-mtp: hook install failed (not SM120, "
              f"ignoring): {exc}", file=sys.stderr, flush=True)

# sm120 NVFP4 DS-MLA patch (independent gate, nvfp4_ds_mla_patch/):
# --kv-cache-dtype nvfp4_ds_mla on the SM120 sparse-MLA backend through our
# cuda-oxide writer/decode kernels (GLM 5.3). Same deferred meta_path hooks
# + fail-closed posture; composes with SUFFIX_SM120_HISPARSE_MTP (disjoint
# target files). Unset = fully inert.
if os.environ.get("SUFFIX_SM120_NVP4DSMLA", "").strip() == "1":
    try:
        from nvfp4_ds_mla_patch import install_post_import_hook as _dsmla_hook
        _dsmla_hook()
    except Exception as exc:
        import sys
        _sm120 = False
        try:
            import torch
            _sm120 = (torch.cuda.is_available()
                      and torch.cuda.get_device_capability()[0] == 12)
        except Exception:
            pass
        if _sm120:
            raise SystemExit(
                f"suffix sm120 nvfp4-ds-mla installation failed on SM120: {exc}"
            ) from exc
        print(f"suffix sm120 nvfp4-ds-mla: hook install failed (not SM120, "
              f"ignoring): {exc}", file=sys.stderr, flush=True)

# In-pod engine-step profiler (suffix_hybrid/step_profiler.py):
# SUFFIX_PROFILE_STEPS=<N>:<skip>[:<per>] profiles N EngineCore steps once
# the load bucket (c1/c2-6/c7-12/c13-24/c25+) held <skip> steps, <per>
# windows per bucket; one "[suffix-prof]" summary block per window.
# Fail-soft: a profiler error never touches serving.
if os.environ.get("SUFFIX_PROFILE_STEPS", "").strip():
    try:
        from suffix_hybrid.step_profiler import (
            install_post_import_hook as _prof_hook)
        _prof_hook()
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        import sys
        print(f"[suffix-prof] hook install failed (profiler off): {exc!r}",
              file=sys.stderr, flush=True)

# cuda-oxide toolchain probe (docs/oxide-kernels.md): SUFFIX_OXIDE_PROBE=1
# driver-loads every bundle cubin (suffix_hybrid/oxide_cubins) and runs the
# probe kernel once per pod, in a CHILD process before vLLM sizes memory.
# Marker in the pod log: "[suffix oxide] OXIDE-PROBE PASS|FAIL". Evidence
# only: kernel lanes fail closed on their own gates.
if (os.environ.get("SUFFIX_OXIDE_PROBE", "").strip() == "1"
        and not os.environ.get("SUFFIX_OXIDE_PROBE_DONE")):
    import subprocess
    import sys
    os.environ["SUFFIX_OXIDE_PROBE_DONE"] = "1"
    subprocess.run([sys.executable, "-m", "suffix_hybrid.oxide_kernels"])

# NVFP4 W4A4 decode-GEMM spike (qwen38-27b-kernels.md §7): SUFFIX_NVFP4_GEMM_
# SPIKE=oracle|bench|both runs `python -m suffix_hybrid.kernels.nvfp4_gemm`
# once per pod in a CHILD process (its CUDA context is gone before vLLM sizes
# memory). Evidence only: markers "[suffix nvfp4-gemm] NVFP4-GEMM ORACLE
# PASS|FAIL" and "bench ..." lines; never gates serving.
_nvfp4_gemm = os.environ.get("SUFFIX_NVFP4_GEMM_SPIKE", "").strip().lower()
if _nvfp4_gemm in ("oracle", "bench", "both") and not os.environ.get(
        "SUFFIX_NVFP4_GEMM_SPIKE_DONE"):
    import subprocess
    import sys
    os.environ["SUFFIX_NVFP4_GEMM_SPIKE_DONE"] = "1"
    subprocess.run([sys.executable, "-m", "suffix_hybrid.kernels.nvfp4_gemm",
                    _nvfp4_gemm])

# Boot gates: the kernel-lab gates that need no model, run once per pod in a
# CHILD process before vLLM sizes memory — how silicon evidence is gathered
# when no lab can start (the lab image comes from a registry; this path only
# needs the stock vLLM image + the bundle). SUFFIX_BOOT_GATES is a comma list
# of ALLOWLISTED names (no free-form modules). Evidence only: never blocks
# serving. Children get the pod env minus other SUFFIX_* feature gates, so the
# pool's own patches never leak into the gate's vLLM instances.
_BOOT_GATES = {
    "nvfp4_dsmla_oracle": (["-m", "nvfp4_ds_mla_patch.oracle"], {}),
    "nvfp4_dsmla_bench": (["-m", "nvfp4_ds_mla_patch.oracle", "--bench", "--json"], {}),
    # 200 GPU blocks (12.8k tokens) < 5 x 4k prompts: forces host spills.
    "hisparse_mtp_oracle": (["-m", "hisparse_mtp_patch.oracle", "--k", "3",
                             "--prompt-len", "4096", "--gpu-blocks", "200"],
                            {"SUFFIX_SM120": "1"}),
}
_boot_gates = [g.strip() for g in os.environ.get("SUFFIX_BOOT_GATES", "").split(",") if g.strip()]
if _boot_gates and not os.environ.get("SUFFIX_BOOT_GATES_DONE"):
    import subprocess
    import sys
    os.environ["SUFFIX_BOOT_GATES_DONE"] = "1"
    for _g in _boot_gates:
        if _g not in _BOOT_GATES:
            print(f"[suffix boot-gate] {_g}: unknown gate, skipped", file=sys.stderr, flush=True)
            continue
        _argv, _extra = _BOOT_GATES[_g]
        _env = {k: v for k, v in os.environ.items()
                if not k.startswith("SUFFIX_") or k.endswith("_DONE")}
        _env.update(_extra)
        print(f"[suffix boot-gate] {_g}: start", file=sys.stderr, flush=True)
        _rc = subprocess.run([sys.executable] + _argv, env=_env).returncode
        print(f"[suffix boot-gate] {_g}: exit {_rc}", file=sys.stderr, flush=True)

# NVFP4-KV pod warmup oracle (gemma-hd512 dossier c.4). The console pins the
# pod command to `vllm serve`, so the on-silicon numerics gate runs here: once
# per pod (the _DONE marker is inherited by every child), in a CHILD process so
# its CUDA context is gone before vLLM sizes memory. Verdict goes to the pod
# log. Fail-closed only when this pod actually serves NVFP4 KV; otherwise a
# failing oracle is evidence, not an outage.
if (os.environ.get("SUFFIX_SM120_NVP4KV_ORACLE", "").strip() == "1"
        and not os.environ.get("SUFFIX_SM120_NVP4KV_ORACLE_DONE")):
    import subprocess
    import sys
    os.environ["SUFFIX_SM120_NVP4KV_ORACLE_DONE"] = "1"
    print("[suffix sm120-nvfp4-kv] oracle: running on-silicon numerics gate",
          file=sys.stderr, flush=True)
    import shlex
    # SUFFIX_SM120_NVP4KV_ORACLE_ARGS: extra oracle CLI, e.g. "--own" (K2-NVFP4
    # cases) or "--bench --shapes 512:16:2" (microbenchmark, exit 0).
    _rc = subprocess.run([sys.executable, "-m", "nvfp4_kv_patch.oracle"]
                         + shlex.split(os.environ.get(
                             "SUFFIX_SM120_NVP4KV_ORACLE_ARGS", ""))
                         ).returncode
    _verdict = {0: "PASS", 1: "FAIL"}.get(_rc, "NOT RUN")
    print(f"[suffix sm120-nvfp4-kv] oracle: exit {_rc} ({_verdict})",
          file=sys.stderr, flush=True)
    if _rc != 0 and os.environ.get("SUFFIX_SM120_NVP4KV", "").strip() == "1":
        raise SystemExit("suffix sm120 nvfp4-kv: on-silicon oracle did not "
                         f"pass (exit {_rc}); refusing to serve NVFP4 KV")

# NOTE: the WARM-START debug endpoint is NO LONGER armed here. The old
# sys.meta_path build_app loader-proxy never attached on the live pod
# (an earlier importer had already loaded vllm.entrypoints.launchers.app,
# short-circuiting find_spec before our finder ran). The route now uses
# vLLM 0.30.0's NATIVE vllm.endpoint_plugins seam: the repo-root module
# suffix_hybrid_warmstart_ep.py is discovered via the dist-info dir
# suffix_hybrid_warmstart_ep-1.0.dist-info/ (entry point group
# vllm.endpoint_plugins), gated by env VLLM_PLUGINS naming
# "suffix_hybrid_warmstart" (loader not called at all when unset) plus the
# plugin's own SUFFIX_HYBRID_WARMSTART=1 belt gate. With both keys off nothing
# here (or anywhere) runs for warm-start.

