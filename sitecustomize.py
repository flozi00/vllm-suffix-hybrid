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
    + SUFFIX_SM120_HISPARSE_PREFETCH=1 -> also prefetch IndexShare follower
                               rows for multi-token decode (perf A/B knob)
    + SUFFIX_SM120_HISPARSE_NO_MIRROR_STAGING=1 -> never allocate the
                               write-only prefill mirror staging buffer
  SUFFIX_SM120_NVP4DSMLA=1  -> arm the deferred SM120 nvfp4_ds_mla sparse-MLA
                               KV patch (our kernels; fail closed on SM120)
  SUFFIX_FP8_DENSE=1        -> arm the deferred FP8 W8A8 dense-linear patch
                               (fp8_dense_patch/; SUFFIX_FP8_DENSE_LAYERS =
                               glob allowlist; fail closed when enabled)
  SUFFIX_NVFP4_DENSE=1       -> same patch, NVFP4 W4A4 mode (vLLM ModelOpt
                               method; SUFFIX_NVFP4_DENSE_LAYERS allowlist,
                               SUFFIX_NVFP4_DENSE_ACT_AMAX overrides)
  SUFFIX_SM120_NVP4KV_ORACLE=1 -> run the NVFP4-KV on-silicon oracle once per
                               pod before serving (fatal only with NVP4KV=1)
  SUFFIX_FICACHE=seed|dump|both -> FlashInfer autotune cache seed/harvest
                               (inert otherwise, degrades on failure)
  SUFFIX_MTP_TUNE=1         -> online idle-time MTP draft-head LoRA tuning
                               (suffix_hybrid/mtp_tune/; logged refusal)
  SUFFIX_PROFILE_STEPS=<N>:<skip>[:<per>] -> in-pod torch.profiler summary
                               ("[suffix-prof]" lines; fail-soft)
  SUFFIX_SAMPLER_WARMUP=0   -> disable the DEFAULT-ON boot JIT of the top-k/
                               top-p sampler kernels (V2 runner; fail-soft)
  SUFFIX_ROCM_AITER_PAD=1   -> ROCm: pass raw MoE padding to AITER fused_moe
                               (vllm#46201; TP1 MXFP4 MoE corruption; fail closed)
  SUFFIX_ROCM_QSA_TOPK_ROWS=1 -> ROCm: QSA indexer top-k in <=384-row chunks (the
                               >384-row kernel needs hostcall = PCIe atomics)
  SUFFIX_ROCM_QSA_MQA=1     -> ROCm: QSA indexer scores only visible columns
                               (suffix_hybrid/kernels/qsa_mqa_rocm.py)
  SUFFIX_ROCM_MXFP4_A16=1   -> ROCm: dense MXFP4 linears at M <= SUFFIX_ROCM_MXFP4_A16_MAX_M
                               (default 32) as one AITER gemm_a16wfp4 launch
  SUFFIX_ROCM_GDN_MTP=1     -> ROCm: GDN MTP-verify core via AITER's strided
                               gated delta rule (suffix_hybrid/kernels/gdn_mtp_rocm.py)
  SUFFIX_ROCM_HC_FUSE=1     -> ROCm: Qwen4Exp HC silu + up GEMM + gate mix in one
                               kernel (suffix_hybrid/kernels/hc_fused_rocm.py);
                               forces VLLM_USE_AOT_COMPILE=0 (stale-artifact guard)
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

# FP8 dense linears (independent gate, fp8_dense_patch/): BF16 attention /
# indexer / shared-expert / dense-MLP / MTP eh_proj linears -> FP8 W8A8 at
# load (per-channel weights, dynamic per-token activations, vLLM's SM120
# cutlass_scaled_mm). Hook on model_loader.utils; the SM120 check and the
# per-shape kernel self-test run at conversion on the worker. Unset = inert.
# SUFFIX_NVFP4_DENSE=1 (same hook): allowlisted BF16 linears -> vLLM ModelOpt
# NVFP4 W4A4 layers (static activation global scale from proven input bounds).
if (os.environ.get("SUFFIX_FP8_DENSE", "").strip() == "1"
        or os.environ.get("SUFFIX_NVFP4_DENSE", "").strip() == "1"
        or os.environ.get("SUFFIX_ACT_AMAX_RECORD", "").strip()):
    try:
        from fp8_dense_patch import install_post_import_hook as _fp8d_hook
        _fp8d_hook()
    except Exception as exc:
        raise SystemExit(f"suffix fp8-dense installation failed: {exc}") from exc

# Hybrid speculator wrap AFTER every sm120 meta_path hook is armed: install_v2
# imports the V2 model runner, which imports the sparse-MLA backends
# (index_group ...). Armed first, those imports run through the patch finders;
# the other way round nvfp4_ds_mla refuses a live module graph (prod glm
# crash 2026-09-27: "index_group imported before the hook armed").
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

# Online idle-time MTP draft-head tuning (suffix_hybrid/mtp_tune/): engine
# idle hook + V2 runner load_model hook (LoRA on the MTP dense linears,
# capture on rank 0). After the wrap so its propose wrapper sits outside
# ours. Optional feature: a failure is a logged refusal, never fatal.
if os.environ.get("SUFFIX_MTP_TUNE", "").strip() == "1":
    try:
        from suffix_hybrid.mtp_tune import install as _mtp_tune_install
        _mtp_tune_install()
    except Exception as exc:  # noqa: BLE001 - optional feature
        import sys
        print(f"[suffix mtp-tune] install failed (tuning off): {exc!r}",
              file=sys.stderr, flush=True)

# ROCm vLLM source patches (suffix_hybrid/rocm_patches.py): each gate rewrites
# one module before its first import; any failure is fatal (fail closed).
if any(os.environ.get(_g, "").strip() == "1"
       for _g in ("SUFFIX_ROCM_AITER_PAD", "SUFFIX_ROCM_QSA_TOPK_ROWS", "SUFFIX_ROCM_QSA_MQA",
                  "SUFFIX_ROCM_MXFP4_A16", "SUFFIX_ROCM_GDN_MTP", "SUFFIX_ROCM_HC_FUSE")):
    from suffix_hybrid.rocm_patches import install_post_import_hook as _rp_hook
    _rp_hook()
# SUFFIX_ROCM_HC_FUSE rewrites Dynamo-traced code. vLLM's AOT-compile artifacts are
# keyed without traced sources and verified against the unpatched files on disk, so
# a gate-off artifact would load into a gate-on pod and serve stock HC silently;
# the classic compile cache hashes every traced file (hc_fused_rocm.py only when on).
if os.environ.get("SUFFIX_ROCM_HC_FUSE", "").strip() == "1":
    os.environ["VLLM_USE_AOT_COMPILE"] = "0"

# Sampler warmup (suffix_hybrid/sampler_warmup.py): DEFAULT ON,
# SUFFIX_SAMPLER_WARMUP=0 disables. Wraps the V2 worker's warmup_kernels to
# compile vLLM's top-k/top-p Triton variants at boot (the V2 Sampler never
# registers them; prod saw "JIT compilation during inference: _topp_sb_*").
# Hook only fires on import of vllm.v1.worker.gpu_worker; failure = log line.
if os.environ.get("SUFFIX_SAMPLER_WARMUP", "1").strip() != "0":
    try:
        from suffix_hybrid.sampler_warmup import (
            install_post_import_hook as _sw_hook)
        _sw_hook()
    except Exception as exc:  # noqa: BLE001 - latency-only feature
        import sys
        print(f"[suffix sampler-warmup] hook install failed (warmup off): "
              f"{exc!r}", file=sys.stderr, flush=True)

# In-pod engine-step profiler (suffix_hybrid/step_profiler.py):
# SUFFIX_PROFILE_STEPS=<N>:<skip>[:<per>] profiles N EngineCore steps once
# the load bucket (c1/c2-6/c7-12/c13-24/c25+) held <skip> steps, <per>
# windows per bucket; one "[suffix-prof]" summary block per window.
# Fail-soft: a profiler error never touches serving.
# Size-routed NCCL all-reduce (suffix_hybrid/nccl_split.py): SUFFIX_NCCL_BANDS /
# SUFFIX_NCCL_SMALL_* (static bands), SUFFIX_NCCL_AUTOTUNE=1, SUFFIX_NCCL_QAR.
# The module decides from its own gates; all unset = inert.
if any(os.environ.get(k, "").strip() for k in ("SUFFIX_NCCL_SMALL_ALGO", "SUFFIX_NCCL_BANDS",
                                               "SUFFIX_NCCL_AUTOTUNE", "SUFFIX_NCCL_QAR")):
    from suffix_hybrid.nccl_split import install_post_import_hook as _ncsplit_hook
    _ncsplit_hook()

if os.environ.get("SUFFIX_PROFILE_WORKER", "").strip():
    try:
        from suffix_hybrid.step_profiler import install_worker_hook
        install_worker_hook()
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        import sys
        print(f"[suffix-prof] worker hook failed: {exc!r}", file=sys.stderr, flush=True)

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
# SPIKE=oracle|bench|both|sweep runs `python -m suffix_hybrid.kernels.nvfp4_gemm`
# once per pod in a CHILD process (its CUDA context is gone before vLLM sizes
# memory). Evidence only: markers "[suffix nvfp4-gemm] NVFP4-GEMM ORACLE
# PASS|FAIL" and "bench ..." lines; never gates serving.
_nvfp4_gemm = os.environ.get("SUFFIX_NVFP4_GEMM_SPIKE", "").strip().lower()
if _nvfp4_gemm in ("oracle", "bench", "both", "sweep") and not os.environ.get(
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
    # K2 NVFP4-KV attention (gemma): --own numerics (q_len>1, masked slots,
    # wide multi-wave verifies) and the lab's K2 microbench preset
    # (kernel-lab/server.py _nvfp4_kv_bench_args defaults + K2_FIXED).
    "nvfp4_kv_oracle_own": (["-m", "nvfp4_kv_patch.oracle", "--own"], {}),
    "nvfp4_kv_bench": (["-m", "nvfp4_kv_patch.oracle", "--bench", "--json",
                        "--shapes", "512:16:2,256:16:8",
                        "--batches", "1,2,4,8,16,32",
                        "--kvs", "1024,4096,16384,65536,131072",
                        "--q-lens", "1,9", "--iters", "20",
                        "--page", "64", "--max-gb", "8"], {}),
    "nvfp4_dsmla_oracle": (["-m", "nvfp4_ds_mla_patch.oracle"], {}),
    "nvfp4_dsmla_bench": (["-m", "nvfp4_ds_mla_patch.oracle", "--bench", "--json"], {}),
    # NVFP4 routed-experts MoE (gemma-4 26B + GLM-5.3 EP/TP8 shapes, one GPU):
    # ours vs vLLM FlashInfer vs f64 spec + per-stage readback; then us/call.
    "nvfp4_moe_oracle": (["-m", "suffix_hybrid.kernels.nvfp4_moe", "oracle"], {}),
    "nvfp4_moe_bench": (["-m", "suffix_hybrid.kernels.nvfp4_moe", "bench"], {}),
    # cold-L2 us of every tune candidate per M (qwen EP0, gemma, GLM EP/TP8)
    # vs FlashInfer -> paste-able SUFFIX_NVFP4_MOE_TUNE + recommended MAX_M.
    "nvfp4_moe_sweep": (["-m", "suffix_hybrid.kernels.nvfp4_moe", "sweep"], {}),
    # FP8 128x128-block routed experts (SUFFIX_FP8_MOE, qwen3.8 MTP draft
    # shape E 512 / EP2 ranks 0+1, H 2560, I 640, top-10; one GPU = one rank):
    # ours vs vLLM Triton vs f64 spec + per-stage readback; then us/call.
    "fp8_moe_oracle": (["-m", "suffix_hybrid.kernels.fp8_moe", "oracle"], {}),
    "fp8_moe_bench": (["-m", "suffix_hybrid.kernels.fp8_moe", "bench"], {}),
    # DSA indexer logits (SUFFIX_SM120_DSA_INDEXER, GLM-5.3 H=32 / DeepSeek
    # H=64 decode native + flattened MTP rows, prefill): ours vs the Triton
    # shim fallback vs f64, top-2048 sets, vLLM persistent_topk on a
    # NaN-poisoned output; then us/call vs Triton + GB/s vs HBM roofline.
    "dsa_indexer_oracle": (["-m", "suffix_hybrid.kernels.dsa_indexer", "oracle"], {}),
    "dsa_indexer_bench": (["-m", "suffix_hybrid.kernels.dsa_indexer", "bench"], {}),
    # NVFP4 W4A4 dense decode GEMM (SUFFIX_NVFP4_GEMM), M <= 64 at the qwen
    # 27b / gemma / qwen-flash TP2 shapes: ours vs exact ref + vLLM FlashInfer
    # (both input routes); bench us/call vs FlashInfer + suggested
    # SUFFIX_NVFP4_GEMM_ROUTE; sweep adds the measured best split per case.
    "nvfp4_gemm_oracle": (["-m", "suffix_hybrid.kernels.nvfp4_gemm", "oracle"], {}),
    "nvfp4_gemm_bench": (["-m", "suffix_hybrid.kernels.nvfp4_gemm", "bench"], {}),
    "nvfp4_gemm_sweep": (["-m", "suffix_hybrid.kernels.nvfp4_gemm", "sweep"], {}),
    # NVFP4 lm_head (SUFFIX_NVFP4_LMHEAD) at the qwen / gemma / GLM-TP8 head
    # shapes: screen vs exact quantized ref + greedy == bf16; then us/call
    # bf16 head vs NVFP4 screen + rescore (CUDA graphs).
    "nvfp4_lmhead_oracle": (["-m", "suffix_hybrid.kernels.nvfp4_lm_head", "oracle"], {}),
    "nvfp4_lmhead_bench": (["-m", "suffix_hybrid.kernels.nvfp4_lm_head", "bench"], {}),
    # --gpu-blocks -1: hot regions + indexer + one request fit, 5 x 4k
    # prompts don't -> host spills while admission still progresses.
    "hisparse_mtp_oracle": (["-m", "hisparse_mtp_patch.oracle", "--k", "3", "5",
                             "--prompt-len", "4096", "--layers", "2",
                             "--gpu-blocks", "-1"],
                            {"SUFFIX_SM120": "1"}),
    # Full GLM stack inside a real vLLM engine (what prod runs, minus TP=8):
    # nvfp4_ds_mla KV + HiSparse + MTP k=5, ref vs patched greedy parity.
    "glm_stack_oracle": (["-m", "hisparse_mtp_patch.oracle", "--k", "5",
                          "--prompt-len", "4096", "--layers", "2",
                          "--gpu-blocks", "-1", "--kv-cache-dtype", "nvfp4_ds_mla"],
                         {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1"}),
    # Same stack at TP=2 (needs 2 GPUs): the shared /dev/shm HiSparse host
    # pool + per-rank cudaHostRegister that prod's TP=8 takes and TP=1 never
    # does (prod crash 2026-09-26: "cudaHostRegister failed: cudaError.???").
    "glm_stack_tp2_oracle": (["-m", "hisparse_mtp_patch.oracle", "--k", "5",
                              "--prompt-len", "4096", "--layers", "2",
                              "--gpu-blocks", "-1", "--kv-cache-dtype", "nvfp4_ds_mla",
                              "--tp", "2"],
                             {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1"}),
    # Same stack at prod shape: 8 layers with IndexShare followers + draft
    # top-k reuse, max_num_seqs=4, 256-token chunks mixed with decodes, one
    # 8-prompt generate(); ref vs patched vs patched+follower-prefetch.
    "glm_stack_multi_oracle": (["-m", "hisparse_mtp_patch.oracle", "--multi",
                                "--k", "5", "--layers", "8", "--index-freq", "4",
                                "--index-offset", "3", "--batched-tokens", "256",
                                "--gpu-blocks", "-1", "--kv-cache-dtype", "nvfp4_ds_mla"],
                               {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1"}),
    # Multi oracle with the mixed-batch fix on (SUFFIX_SM120_HISPARSE_MIXED):
    # resident prefill slices next to decodes skip the full-context staging.
    "glm_stack_mixed_oracle": (["-m", "hisparse_mtp_patch.oracle", "--multi",
                                "--k", "5", "--layers", "8", "--index-freq", "4",
                                "--index-offset", "3", "--batched-tokens", "256",
                                "--gpu-blocks", "-1", "--kv-cache-dtype", "nvfp4_ds_mla"],
                               {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1",
                                "SUFFIX_SM120_HISPARSE_MIXED": "1"}),
    # Mixed oracle without the write-only prefill mirror staging buffer
    # (SUFFIX_SM120_HISPARSE_NO_MIRROR_STAGING): patched == ref, same spills.
    "glm_stack_nostage_oracle": (["-m", "hisparse_mtp_patch.oracle", "--multi",
                                  "--k", "5", "--layers", "8", "--index-freq", "4",
                                  "--index-offset", "3", "--batched-tokens", "256",
                                  "--gpu-blocks", "-1", "--kv-cache-dtype", "nvfp4_ds_mla"],
                                 {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1",
                                  "SUFFIX_SM120_HISPARSE_MIXED": "1",
                                  "SUFFIX_SM120_HISPARSE_NO_MIRROR_STAGING": "1"}),
    # Same as glm_stack_multi_oracle with synchronous launches: the step-55
    # cudaErrorIllegalAddress (2026-09-27) surfaced asynchronously at the host
    # mirror sync; blocking launches name the faulting kernel.
    "glm_stack_multi_debug": (["-m", "hisparse_mtp_patch.oracle", "--multi", "--eager",
                               "--k", "5", "--layers", "8", "--index-freq", "4",
                               "--index-offset", "3", "--batched-tokens", "256",
                               "--gpu-blocks", "-1", "--kv-cache-dtype", "nvfp4_ds_mla"],
                              {"SUFFIX_SM120": "1", "SUFFIX_SM120_NVP4DSMLA": "1",
                               "CUDA_LAUNCH_BLOCKING": "1", "TORCH_SHOW_CPP_STACKTRACES": "1"}),
    # All-reduce latency on this node's GPUs: NCCL vs vLLM custom AR forced
    # past its NVLink-only gate (prod glm TP=8 PCIe: NCCL all-reduce = 31 % of
    # a decode step). Second entry: same with NCCL's tree algorithm.
    # FP8 dense linears (SUFFIX_FP8_DENSE) at GLM-5.3 TP8 per-rank shapes:
    # FP8 vs BF16 / exact-dequant error per layer type + us/call BF16 vs FP8
    # at M 6..192 (CUDA graphs, L2-busting weight rotation).
    "fp8_dense_oracle": (["-m", "fp8_dense_patch.oracle"], {}),
    # NVFP4 dense linears (SUFFIX_NVFP4_DENSE) at Gemma-4-26B-A4B (TP1) and
    # Gemma-4-31B (TP2) per-rank shapes: NVFP4 vs BF16 / exact-dequant error,
    # activation-headroom sweep, us/call BF16 vs NVFP4 vs FP8 at M 1..64.
    "nvfp4_dense_oracle": (["-m", "fp8_dense_patch.nvfp4_oracle"], {}),
    "allreduce_bench": (["-m", "suffix_hybrid.tools.ar_bench"], {}),
    "allreduce_bench_tree": (["-m", "suffix_hybrid.tools.ar_bench"], {"NCCL_ALGO": "Tree"}),
    # Size-routed all-reduce evidence: default comm vs a second comm built under
    # NCCL_ALGO=allreduce:tree, 12 KiB .. 24 MiB (nccl_split crossover).
    "allreduce_split_bench": (["-m", "suffix_hybrid.tools.ar_bench", "--split", "allreduce:tree",
                               "--sizes-kib", "48,192,768,1536"], {}),
    "allreduce_split_bench_simple": (["-m", "suffix_hybrid.tools.ar_bench", "--split",
                                      "allreduce:ring/Simple", "--sizes-kib", "48,192,768,1536"], {}),
    "allreduce_split_bench_tree_simple": (["-m", "suffix_hybrid.tools.ar_bench", "--split",
                                           "allreduce:tree/Simple", "--sizes-kib", "48,192,768,1536"], {}),
    # Every SUFFIX_NCCL_AUTOTUNE candidate x size (16 KiB..128 MiB + GLM tokens x
    # 6144 x 2), same timing code as the autotune, + the bands it would choose.
    "allreduce_matrix_bench": (["-m", "suffix_hybrid.tools.ar_bench", "--matrix",
                                "--sizes-kib", "48"], {}),
    # Same + nccl_qar int8/fp8 compressed all-reduce vs best NCCL per size.
    "allreduce_qar_bench": (["-m", "suffix_hybrid.tools.ar_bench", "--qar",
                             "--sizes-kib", "48"], {}),
    # ROCm card facts: arch/CUs, AITER tuned-config coverage for this CU count,
    # HBM bandwidth, BF16 vs MXFP4 GEMM at the qwen3.8-flash TP1 decode shapes.
    "rocm_probe": (["-m", "suffix_hybrid.tools.rocm_probe"], {}),
    # One MXFP4 GEMM path per process (HIP faults are sticky): Triton, Triton
    # with a sync after the activation quant, ASM; blocking launches pin the
    # failing kernel.
    "rocm_fp4_triton": (["-m", "suffix_hybrid.tools.rocm_probe", "--only", "triton",
                         "--m", "1,16"], {"HIP_LAUNCH_BLOCKING": "1"}),
    "rocm_fp4_triton_sync": (["-m", "suffix_hybrid.tools.rocm_probe", "--only", "triton-sync",
                              "--m", "1"], {"HIP_LAUNCH_BLOCKING": "1"}),
    "rocm_fp4_asm": (["-m", "suffix_hybrid.tools.rocm_probe", "--only", "asm",
                      "--m", "1,16"], {"HIP_LAUNCH_BLOCKING": "1"}),
    # Host/KFD/torch device limits + one Triton matmul per LDS size, each in
    # its own process: which kernels launch on this card and runtime.
    "rocm_lds_probe": (["-m", "suffix_hybrid.tools.rocm_lds_probe"], {"AMD_LOG_LEVEL": "1"}),
    # Same host/KFD facts (incl. io_link atomics flags) without the kernel matrix.
    "rocm_host_facts": (["-m", "suffix_hybrid.tools.rocm_lds_probe", "--facts"], {}),
    # SUFFIX_ROCM_QSA_MQA kernel vs vLLM's qsa_mqa_paged: equality on visible columns + us/call.
    "qsa_mqa_bench": (["-m", "suffix_hybrid.kernels.qsa_mqa_rocm"], {}),
    # SUFFIX_ROCM_GDN_MTP vs vLLM's spec branch of _forward_core_rocm: outputs, every
    # state page byte, graph replay with new slots + graphed us/call (MTP-4 verify).
    "gdn_mtp_bench": (["-m", "suffix_hybrid.kernels.gdn_mtp_rocm"], {}),
    # SUFFIX_ROCM_HC_FUSE kernel vs vLLM hc_silu -> F.linear -> hc_gate_mix: bf16
    # bound + bit-exact share, us/call (HIP graphs, cold weights), BMxNG sweep.
    "hc_fuse_bench": (["-m", "suffix_hybrid.kernels.hc_fused_rocm"], {}),
    # AITER fused-MoE tuner for this card's CU count (qwen3.8-flash MXFP4 MoE; the
    # _fse variant = shared expert fused as expert 513, top-11); prints the CSV.
    "aiter_moe_tune": (["-m", "suffix_hybrid.tools.aiter_moe_tune"], {}),
    "aiter_moe_tune_fse": (["-m", "suffix_hybrid.tools.aiter_moe_tune", "--expert", "513",
                            "--topk", "11"], {}),
    # MXFP4 lm_head (SUFFIX_MXFP4_LMHEAD) at the qwen3.8-flash head: fidelity +
    # us/call of the graphed screen+rescore vs the stock bf16 head, M=1..16.
    "mxfp4_lmhead_bench": (["-m", "suffix_hybrid.kernels.mxfp4_lm_head"], {}),
    # SUFFIX_ROCM_MXFP4_A16 at the qwen3.8-flash dense MXFP4 shapes, M 1..40: gemm_a16wfp4
    # vs vLLM's quant + gemm_afp4wfp4 (numerics, graph replay == eager), graphed us/call.
    "mxfp4_a16_bench": (["-m", "suffix_hybrid.kernels.mxfp4_a16_rocm"], {}),
}
_boot_gates = [g.strip() for g in os.environ.get("SUFFIX_BOOT_GATES", "").split(",") if g.strip()]
def _boot_gates_claim():
    # Once per POD: the env marker only reaches children, but `vllm serve`
    # starts more than one top-level Python (prod glm 2026-09-27 ran every
    # gate twice) -> an O_EXCL marker file in the pod's /tmp decides.
    try:
        os.close(os.open(os.environ.get("SUFFIX_BOOT_GATES_MARKER", "/tmp/suffix_boot_gates.claimed"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False
    except OSError:
        return True


if (_boot_gates and not os.environ.get("SUFFIX_BOOT_GATES_DONE")
        and _boot_gates_claim()):
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

