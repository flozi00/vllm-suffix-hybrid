# SPDX-License-Identifier: Apache-2.0
"""Why do Triton launches fail with HIP 401 on the MI350P? (boot gate rocm_lds_probe)

The first MI350P boot failed every large Triton kernel (AITER MXFP4 GEMM, FLA
GDN prefill) with hipErrorIllegalState while small Triton kernels, hipBLASLt
and AITER's prebuilt CK kernels ran. Hypothesis: the runtime reports less LDS
than gfx950 has (160 KiB), so kernels above the reported limit cannot launch.

Prints "[suffix lds-probe]" lines: amdgpu driver version, the KFD topology's
LDS / feature properties, torch's device limits, then one CHILD process per
Triton matmul config (a 401 is sticky for the process) with the kernel's
compiled shared-memory size and whether the launch succeeded.

Every failing config of the first run had 2 pipeline stages, which on gfx950
also turns on Triton's direct-to-LDS async copies. So each config runs in three
modes: default; noasync (TRITON_HIP_USE_ASYNC_COPY=0); cap64k (rocm_lds_cap
refits to <= 64 KiB). Size-only failures pass in cap64k; async-copy failures
also pass in noasync, and fail even for small pipelined tiles.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

MARK = "[suffix lds-probe]"
# (BLOCK_M, BLOCK_N, BLOCK_K, num_stages): rising LDS footprint
CONFIGS = [(64, 64, 32, 2), (128, 128, 32, 2), (128, 128, 64, 1), (128, 128, 64, 2),
           (128, 256, 128, 1), (256, 256, 128, 1), (128, 128, 128, 2)]
MODES = {"default": {}, "noasync": {"TRITON_HIP_USE_ASYNC_COPY": "0"},
         "cap64k": {"SUFFIX_ROCM_LDS_CAP": "65536"}}


def say(msg: str) -> None:
    print(f"{MARK} {msg}", flush=True)


def host_facts() -> None:
    for path in ("/sys/module/amdgpu/version", "/proc/version"):
        try:
            say(f"{path}: {open(path).read().strip()[:200]}")
        except OSError as exc:
            say(f"{path}: {exc}")
    for props in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties")):
        try:
            kv = dict(line.split() for line in open(props) if len(line.split()) == 2)
        except OSError as exc:
            say(f"{props}: {exc}")
            continue
        if kv.get("simd_count", "0") == "0":
            continue  # CPU node
        keep = ("lds_size_in_kb", "local_mem_size", "gfx_target_version", "simd_count",
                "simd_per_cu", "max_waves_per_simd", "cu_per_simd_array", "array_count",
                "num_xcc", "capability", "capability2", "device_id", "fw_version",
                "sdma_fw_version", "max_engine_clk_fcompute", "debug_prop")
        say(f"kfd {props.split('/')[-2]}: " + " ".join(f"{k}={kv[k]}" for k in keep if k in kv))
    import torch
    p = torch.cuda.get_device_properties(0)
    fields = {k: getattr(p, k) for k in dir(p) if not k.startswith("_")
              and isinstance(getattr(p, k), (int, str, float, bool))}
    say("torch props: " + json.dumps(fields, default=str)[:1500])


def one(cfg, mode: str) -> None:
    """Child: compile + launch one bf16 tl.dot matmul config, report LDS + outcome."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm, pn = tl.program_id(0), tl.program_id(1)
        rm, rn, rk = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN), tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, K, BK):
            x = tl.load(a + rm[:, None] * K + (k + rk)[None, :])
            y = tl.load(b + (k + rk)[:, None] * N + rn[None, :])
            acc += tl.dot(x, y)
        tl.store(c + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16))

    bm, bn, bk, st = cfg
    m = n = k = 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    grid = (m // bm, n // bn)
    try:
        h = mm.warmup(a, b, c, m, n, k, BM=bm, BN=bn, BK=bk, num_stages=st, grid=grid)
        shared = f"{getattr(h.metadata, 'shared', '?')} B/{getattr(h.metadata, 'num_stages', '?')}st"
    except Exception as exc:  # noqa: BLE001
        say(f"{mode} cfg {cfg}: COMPILE FAILED {exc!r}"[:300])
        return
    try:
        mm[grid](a, b, c, m, n, k, BM=bm, BN=bn, BK=bk, num_stages=st)
        torch.cuda.synchronize()
        err = ((c.float() - (a.float() @ b.float())).norm() / (a.float() @ b.float()).norm()).item()
        say(f"{mode} cfg {cfg}: shared {shared} -> LAUNCH OK (rel {err:.1e})")
    except Exception as exc:  # noqa: BLE001
        say(f"{mode} cfg {cfg}: shared {shared} -> LAUNCH FAILED {exc!r}"[:300])


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(tuple(int(v) for v in sys.argv[2].split(",")), sys.argv[3])
        return 0
    host_facts()
    for mode, extra in MODES.items():
        for cfg in CONFIGS:
            subprocess.run([sys.executable, "-m", "suffix_hybrid.tools.rocm_lds_probe", "--one",
                            ",".join(map(str, cfg)), mode], env={**os.environ, **extra}, timeout=600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
