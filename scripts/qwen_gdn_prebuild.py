#!/usr/bin/env python3
"""CI: prebuild the K-GDN1 cubins for sm_120 (no GPU needed).

`tileiras` (pip `nvidia-cuda-tileiras`) is an offline Tile IR -> cubin
compiler: file in, file out, target given by `--gpu-name` (cutile-compiler
cuda_tile_runtime_utils.rs `run_tileiras`); its ELF links only libc/libm/
libpthread/librt/libdl and has no `cuInit`/driver dependency. So the runner
compiles every serving specialization (act x div(T) x div(S)) of the
qwen3.8-27b shape and writes the cubins + manifest.json into DIR, which
scripts/runtime_bundle.py ships next to the .so (suffix_hybrid/qgdn_cubins/).
At pod startup (SUFFIX_QWEN_GDN=1) suffix_hybrid.kernels.qwen_gdn verifies the
manifest and installs the cubins; no JIT ever runs on a pod.

Needs: the qwen-gdn-kernels wheel installed, CUTILE_TILEIRAS_PATH set.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

# Served shapes: (H, HV, K) — qwen3.8-27b-fable-distill (Qwen3.5-27B GDN).
SHAPES = [(16, 48, 128)]
ACTS = (0, 1)  # silu, sigmoid
DIVS = (1, 2, 4, 8, 16)
ARCH = "sm_120"
BYTECODE_VERSION = "13.2"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("CUTILE_BYTECODE_VERSION", BYTECODE_VERSION)
    if os.environ["CUTILE_BYTECODE_VERSION"] != BYTECODE_VERSION:
        raise SystemExit("CUTILE_BYTECODE_VERSION must be " + BYTECODE_VERSION)
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if not tileiras or not os.path.isfile(tileiras):
        raise SystemExit("CUTILE_TILEIRAS_PATH must point at the tileiras binary")
    from suffix_hybrid import _native as native
    if not getattr(native, "HAS_QWEN_GDN_CUDA", False):
        raise SystemExit("wheel built without cargo feature qwen-gdn-kernels")

    version = subprocess.run([tileiras, "--version"], capture_output=True,
                             text=True, check=True).stdout.strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for h, hv, k in SHAPES:
        for act in ACTS:
            for t in DIVS:
                for s in DIVS:
                    bc, ver, bc_sha = native.qwen_gdn_variant_bytecode(
                        h, hv, k, act, t, s, ARCH)
                    if ver != BYTECODE_VERSION:
                        raise SystemExit(f"bytecode version {ver} != {BYTECODE_VERSION}")
                    cubin = native.qwen_gdn_compile_cubin(bc, ARCH)
                    name = f"k_gdn1_h{h}_hv{hv}_k{k}_act{act}_t{t}_s{s}.cubin"
                    (out / name).write_bytes(cubin)
                    entries.append({
                        "file": name, "h": h, "hv": hv, "k": k, "act": act,
                        "t_div": t, "s_div": s, "bc_sha256": bc_sha,
                        "sha256": hashlib.sha256(cubin).hexdigest(),
                        "bytes": len(cubin),
                    })
    manifest = {
        "kernel": "gdn_decode_fused_k1",
        "arch": ARCH,
        "tileiras_version": version,
        "bytecode_version": BYTECODE_VERSION,
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"K-GDN1 prebuild: {len(entries)} cubins for {ARCH} -> {out}")


if __name__ == "__main__":
    main()
