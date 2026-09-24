#!/usr/bin/env python3
"""CI diagnostics for tileiras rejections (K-GDN1 prebuild; reusable by K2).

tileiras 13.4.92 reports only "error: failed to compile Tile IR program" for
a rejected program. This prints tileiras' OWN raw stdout/stderr (never the
cutile wrapper's summary) for:
  1. `--version` and `--list-versions` (supported bytecode versions);
  2. the failing bytecode (if given) at -O3 and -O0, plus remark flags;
  3. construct probes from the wheel (`_native.qwen_gdn_probe_bytecodes`):
     tile-size sweep, partition_full_mut dynamic store, if/else stores, bf16
     I/O, [1,1] scalar math, scalar broadcast, and K-GDN1 at K = 32/64/128;
  4. the K-GDN1 K=128 program re-serialized at every other bytecode version
     tileiras lists (answers "is the CUTILE_BYTECODE_VERSION pin the cause?").
Ends with a PASS/FAIL table. Exit status is 0 (diagnostics only).

Usage: python scripts/tileiras_diag.py [--bc FAILING.bc] [--gpu sm_120]
Needs: CUTILE_TILEIRAS_PATH, the qwen-gdn-kernels wheel.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile


def run(tileiras, args, label):
    p = subprocess.run([tileiras, *args], capture_output=True, text=True)
    print(f"--- tileiras {' '.join(args)}  [{label}] exit={p.returncode}")
    for stream in (p.stdout, p.stderr):
        if stream.strip():
            print(stream.rstrip())
    return p.returncode


def compile_bc(tileiras, bc, gpu, label, extra=()):
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "p.bc"), os.path.join(d, "p.cubin")
        with open(src, "wb") as f:
            f.write(bc)
        rc = run(tileiras, ["--gpu-name", gpu, *extra, "-o", dst, src], label)
        return rc == 0 and os.path.getsize(dst) > 0 if os.path.exists(dst) else False


_VER_CHILD = r"""
import os, sys
from suffix_hybrid import _native as n
bc = dict(n.qwen_gdn_probe_bytecodes(sys.argv[1]))["k_gdn1_K128"]
sys.stdout.buffer.write(bc)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", help="failing bytecode file to re-run verbosely")
    ap.add_argument("--gpu", default="sm_120")
    a = ap.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH", "tileiras")
    print("=" * 30, "tileiras diagnostics", "=" * 30)
    run(tileiras, ["--version"], "version")
    run(tileiras, ["--list-versions"], "bytecode versions")
    results = []
    if a.bc:
        bc = open(a.bc, "rb").read()
        for extra, lab in ((("--opt-level", "3"), "failing -O3"),
                           (("--opt-level", "0"), "failing -O0"),
                           (("--remarks-failed=all",), "failing +remarks-failed"),
                           (("--remark-format=text", "--remarks=all"), "failing +remarks")):
            results.append((lab, compile_bc(tileiras, bc, a.gpu, lab, extra)))
    from suffix_hybrid import _native as n
    for name, bc in n.qwen_gdn_probe_bytecodes(a.gpu):
        results.append((name, compile_bc(tileiras, bytes(bc), a.gpu, name)))
    # bytecode-version question: re-serialize K-GDN1 at each listed version
    lv = subprocess.run([tileiras, "--list-versions"], capture_output=True, text=True)
    versions = sorted(set(re.findall(r"\b(1[3-9]\.\d+)\b", lv.stdout + lv.stderr)))
    for ver in versions:
        env = dict(os.environ, CUTILE_BYTECODE_VERSION=ver)
        # cwd outside the checkout: its source suffix_hybrid/ (no .so) would
        # shadow the installed wheel (d057d457 CI: every sweep row failed so).
        p = subprocess.run([sys.executable, "-c", _VER_CHILD, a.gpu], env=env,
                           capture_output=True, cwd=tempfile.gettempdir())
        if p.returncode != 0:
            print(f"--- serialize K-GDN1 at bytecode {ver} failed:\n"
                  f"{p.stderr.decode(errors='replace')[-2000:]}")
            results.append((f"k_gdn1_K128@bc{ver}", False))
            continue
        lab = f"k_gdn1_K128@bc{ver}"
        results.append((lab, compile_bc(tileiras, p.stdout, a.gpu, lab)))
    print("=" * 30, "tileiras probe table", "=" * 30)
    for lab, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
