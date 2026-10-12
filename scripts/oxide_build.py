#!/usr/bin/env python3
"""CI: build every cuda-oxide kernel crate into an sm_120 cubin (no GPU).

Convention (docs/oxide-kernels.md): each directory kernels-oxide/<name>/ is a
standalone cargo crate (own [workspace]) whose #[cuda_module] kernels
cargo-oxide compiles to PTX. Per crate:

  cargo +<nightly> oxide build --arch sm_120      -> <name>.ptx
  assert  .target sm_120  and  .version <= MAX_PTX_ISA (CUDA 13.0 floor)
  ptxas -arch=sm_120 -O3 <name>.ptx -o <name>.cubin   (ptxas MUST be 13.0)
  cuobjdump --list-elf <name>.cubin  must list sm_120 SASS

and writes OUT/<name>.cubin + OUT/manifest.json (ptxas version, PTX ISA,
entries, sha256), which scripts/runtime_bundle.py ships as
suffix_hybrid/oxide_cubins/. Pods load the SASS directly (the vLLM image has
no PTX JIT library and driver 580 rejects newer-toolchain SASS: silicon
verdict 2026-09-25), so nothing here may produce PTX-only artifacts.

Env: OXIDE_NIGHTLY (default nightly-2026-08-28), OXIDE_PTXAS, OXIDE_CUOBJDUMP,
CUDA_OXIDE_REV (recorded in the manifest).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ARCH = "sm_120"
ALLOWED_ARCHS = ("sm_120", "sm_120a")
MAX_PTX_ISA = (9, 0)  # CUDA 13.0 ptxas accepts PTX ISA <= 9.0
PTXAS_RELEASE = "13.0"
ROOT = Path(__file__).resolve().parents[1]


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def ptx_facts(ptx: str):
    ver = re.search(r"^\.version (\d+)\.(\d+)", ptx, re.M)
    tgt = re.search(r"^\.target (\S+)", ptx, re.M)
    sigs = re.findall(r"^\.visible \.entry (\w+)\(([^)]*)\)", ptx, re.M)
    if not ver or not tgt:
        raise SystemExit("PTX without .version/.target")
    # param counts: the host checks them at load (a stale cubin fails closed)
    params = {e: body.count(".param") for e, body in sigs}
    return (int(ver.group(1)), int(ver.group(2))), tgt.group(1), [e for e, _ in sigs], params


def build_one(crate, var, nightly, ptxas, cuobjdump, out):
    name = crate.name + var["suffix"]
    # Per-variant target: "sm_120" (default) or the arch-specific "sm_120a"
    # (needed by e.g. block-scaled FP4 mma kind::mxf4nvf4; the driver loads
    # sm_120a SASS on cc 12.0 devices only).
    arch = var.get("arch", ARCH)
    if arch not in ALLOWED_ARCHS:
        raise SystemExit(f"{name}: arch {arch} not in {ALLOWED_ARCHS}")
    ptx_path = crate / f"{crate.name}.ptx"
    if ptx_path.exists():
        ptx_path.unlink()
    cmd = ["cargo", f"+{nightly}", "oxide", "build", "--arch", arch]
    if var["features"]:
        cmd += ["--features", var["features"]]
    run(cmd, cwd=crate)
    if not ptx_path.is_file():
        raise SystemExit(f"{name}: cargo-oxide produced no {ptx_path.name}")
    ptx = ptx_path.read_text()
    isa, target, entries, params = ptx_facts(ptx)
    if target != arch or isa > MAX_PTX_ISA or not entries:
        raise SystemExit(f"{name}: .target {target} .version {isa} entries {entries} "
                         f"(need {arch}, <= {MAX_PTX_ISA}, >= 1 entry)")
    if ".local" in ptx:
        print(f"WARNING {name}: PTX uses local memory (register arrays spilled)")
    cubin = out / f"{name}.cubin"
    run([ptxas, f"-arch={arch}", "-O3", *var["ptxas"], "-o", str(cubin), str(ptx_path)])
    elf = run([cuobjdump, "--list-elf", str(cubin)]).stdout
    if ARCH not in elf:
        raise SystemExit(f"{name}: cubin has no {ARCH} SASS:\n{elf}")
    print(run([cuobjdump, "--dump-resource-usage", str(cubin)]).stdout, flush=True)
    (out / f"{name}.ptx").write_text(ptx)  # audit copy (never loaded)
    data = cubin.read_bytes()
    return {
        "name": name,
        "file": cubin.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "ptx_isa": f"{isa[0]}.{isa[1]}",
        "ptxas_flags": var["ptxas"],
        "arch": arch,
        "entries": entries,
        "params": params,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernels", default=str(ROOT / "kernels-oxide"))
    ap.add_argument("--only", default="", help="comma list of crate names")
    args = ap.parse_args()
    nightly = os.environ.get("OXIDE_NIGHTLY", "nightly-2026-08-28")
    ptxas = os.environ.get("OXIDE_PTXAS", "ptxas")
    cuobjdump = os.environ.get("OXIDE_CUOBJDUMP", "cuobjdump")

    pv = run([ptxas, "--version"]).stdout
    m = re.search(r"release (\d+\.\d+), V([\d.]+)", pv)
    if not m or m.group(1) != PTXAS_RELEASE:
        raise SystemExit(f"ptxas must be CUDA {PTXAS_RELEASE}, got:\n{pv}")
    ptxas_version = m.group(2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {x for x in args.only.split(",") if x}
    kernels, failed = [], []
    for crate in sorted(Path(args.kernels).iterdir()):
        if not (crate / "Cargo.toml").is_file() or (only and crate.name not in only):
            continue
        # Optional register/shape variants: one cubin per entry, built with
        # its cargo features and extra ptxas flags, named <crate><suffix>.
        vf = crate / "oxide-variants.json"
        variants = (json.loads(vf.read_text()) if vf.is_file()
                    else [{"suffix": "", "features": "", "ptxas": []}])
        for var in variants:
            # Per-cubin isolation: one crate's toolchain failure must not
            # drop every other lane's cubins from the bundle (lanes whose
            # cubin is missing fail closed at ensure_loaded). Still exit 1.
            try:
                kernels.append(build_one(crate, var, nightly, ptxas, cuobjdump, out))
            except (SystemExit, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or exc
                print(f"FAILED {crate.name}{var['suffix']}: {detail}", flush=True)
                failed.append(crate.name + var["suffix"])
    if not kernels:
        raise SystemExit(f"no kernel cubins built (failed: {failed})")
    manifest = {
        "arch": ARCH,
        "ptxas_version": ptxas_version,
        "cuda_oxide_rev": os.environ.get("CUDA_OXIDE_REV", "unknown"),
        "nightly": nightly,
        "kernels": kernels,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"oxide build: {len(kernels)} cubins ({ARCH}, ptxas {ptxas_version}) -> {out}")
    if failed:
        raise SystemExit(f"oxide build: FAILED {failed} (manifest lists the rest)")


if __name__ == "__main__":
    main()
