#!/usr/bin/env python3
"""CI: prebuild the K2-NVFP4 attention cubins for sm_120 (no GPU needed).

Same contract as scripts/qwen_gdn_prebuild.py: the offline `tileiras`
compiles every serving specialization; scripts/runtime_bundle.py ships the
cubins + manifest.json as suffix_hybrid/nvfp4_attn_cubins/; at pod startup
(SUFFIX_SM120_NVP4KV_OWN_ATTN=1) sm120/nvfp4_kv_patch/own_attn.py verifies the
manifest, rebuilds each variant's bytecode (GPU-free) and installs the cubins
— no JIT ever runs on a pod.

Variant = served shape (head_dim, q heads, kv heads, page size) x q_len x the
window_left divisibility class x split count NS (every NS the launch plan can
pick for batch 1..--max-batch). Everything else in the cutile key is made
shape-constant by the launch layout (src/nvfp4_attn_gpu.rs).

--served entries: d:hq:hkv:page:q_lens:window_left, q_lens "1-9" or "1,9".
Defaults: gemma-4 full-attn + SWA at page 16 (serving, q_len 1..9 for MTP
k=8 / suffix widths), the same at page 64 (oracle), qwen3.8-27b at page 2816
(serving) and 64 (oracle). A pool whose shape is missing refuses to start and
names the entry to add.

Needs: the nvfp4-attn-kernels wheel installed, CUTILE_TILEIRAS_PATH set.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

SERVED = [
    "512:16:2:16:1-9:-1",
    "256:16:8:16:1-9:1023",
    "512:16:2:64:1,9:-1",
    "256:16:8:64:1,9:1023",
    "256:24:4:2816:1:-1",
    "256:24:4:64:1:-1",
]
KERNELS = ("nvfp4_attn_partial", "nvfp4_attn_merge")
ARCH = "sm_120"
BYTECODE_VERSION = "13.2"
NUM_SMS = 188  # RTX PRO 6000 Blackwell; the pod plan uses its own count


def q_lens(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--served", action="append", default=None)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--num-sms", type=int, default=NUM_SMS)
    args = ap.parse_args()
    os.environ.setdefault("CUTILE_BYTECODE_VERSION", BYTECODE_VERSION)
    if os.environ["CUTILE_BYTECODE_VERSION"] != BYTECODE_VERSION:
        raise SystemExit("CUTILE_BYTECODE_VERSION must be " + BYTECODE_VERSION)
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if not tileiras or not os.path.isfile(tileiras):
        raise SystemExit("CUTILE_TILEIRAS_PATH must point at the tileiras binary")
    from suffix_hybrid import _native as native
    if not getattr(native, "HAS_NVFP4_ATTN_CUDA", False):
        raise SystemExit("wheel built without cargo feature nvfp4-attn-kernels")

    version = subprocess.run([tileiras, "--version"], capture_output=True,
                             text=True, check=True).stdout.strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    entries, built = [], {}
    for spec in args.served or SERVED:
        d, hq, hkv, page, ql, wl = spec.split(":")
        d, hq, hkv, page, wl = int(d), int(hq), int(hkv), int(page), int(wl)
        for q_len in q_lens(ql):
            # The pod plan uses the live SM count; 188 is the only SM120 part
            # we serve on. A different count picks other NS -> refused loudly.
            for ns in native.nvfp4_attn_split_set(d, hq, hkv, page, q_len,
                                                  args.num_sms, args.max_batch):
                for kernel in KERNELS:
                    bc, ver, bc_sha = native.nvfp4_attn_variant_bytecode(
                        d, hq, hkv, page, q_len, wl, ns, kernel, ARCH)
                    if ver != BYTECODE_VERSION:
                        raise SystemExit(f"bytecode version {ver} != {BYTECODE_VERSION}")
                    # identical bytecode (e.g. merge across page sizes)
                    # compiles once
                    cubin = built.get(bc_sha)
                    if cubin is None:
                        cubin = built[bc_sha] = native.nvfp4_attn_compile_cubin(
                            bc, ARCH)
                    name = (f"{kernel}_d{d}_hq{hq}_hkv{hkv}_p{page}_q{q_len}"
                            f"_ns{ns}.cubin")
                    (out / name).write_bytes(cubin)
                    entries.append({
                        "file": name, "kernel": kernel, "d": d, "hq": hq,
                        "hkv": hkv, "page": page, "q_len": q_len,
                        "window_left": wl, "ns": ns, "bc_sha256": bc_sha,
                        "sha256": hashlib.sha256(cubin).hexdigest(),
                        "bytes": len(cubin),
                    })
    manifest = {
        "kernel": "k2_nvfp4_attn",
        "arch": ARCH,
        "num_sms": args.num_sms,
        "tileiras_version": version,
        "bytecode_version": BYTECODE_VERSION,
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"K2-NVFP4 prebuild: {len(entries)} variants ({len(built)} unique "
          f"tileiras compiles) for {ARCH} -> {out}")


if __name__ == "__main__":
    main()
