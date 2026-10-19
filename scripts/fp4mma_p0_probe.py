#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P0 probe of the native-FP4-MMA family: does Triton lower `tl.dot_scaled` (e2m1 x e2m1, e4m3
block-16 scales) to `mma.sync...kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3` for sm_120a, for
EXACTLY the load / scale-unpack patterns our Helion bodies emit, and which tile configs are legal?

LINUX ONLY, NO GPU NEEDED (Triton has no macOS wheel). NOT RUN by its author (a Mac): every path below that
touches Triton is unrun code; `--selftest` (no Triton) runs here and in CI. Run it once per wheel, each in
its own venv (the two versions cannot share an environment):

    python -m venv /tmp/v371 && /tmp/v371/bin/pip install triton==3.7.1 && \\
        PYTHONPATH=<checkout> /tmp/v371/bin/python scripts/fp4mma_p0_probe.py --quick --out p0-3.7.1.json
    python -m venv /tmp/v380 && /tmp/v380/bin/pip install triton==3.8.0 && \\
        PYTHONPATH=<checkout> /tmp/v380/bin/python scripts/fp4mma_p0_probe.py --quick --out p0-3.8.0.json
    python scripts/fp4mma_p0_probe.py --compare p0-3.7.1.json p0-3.8.0.json

How it compiles (read from the 3.7.1 / 3.8.0 sources): `triton.compile(ASTSource(fn, signature,
constexprs, attrs), target=GPUTarget("cuda", 120, 32), options=...)` needs no driver and no GPU: the
wheel's bundled ptxas-blackwell assembles the PTX (`.target sm_120a`). This is the AOT path of
triton/tools/compile.py minus `create_binder()` (the only call that wants a driver). The fake-driver
technique of the 2026-10-11 session (memory custom-kernels-always) is only needed to push a HELION-
GENERATED host function through `JITFunction.run`: `--helion` does that (a duck-typed active driver +
`knobs.runtime.jit_cache_hook` that compiles for the fake target and returns True so nothing launches),
on CPU tensors, for the real bodies of nvfp4_gemm_fp4mma.py / dsv41_moe_fp4mma.py.

Per variant x config it asserts (fp4mma_probe.ptx_problems with allow_local=False): `.target sm_120a`,
the instruction `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.
e2m1.f32.ue4m3`, no `.local` (spills / stack), smem <= 101376 B; and prints registers / spill bytes /
smem from `ptxas -v` on the PTX. Variants (the mirror kernel is the PR-38338 kernel shape plus the
patterns our Helion lowering produced, see docs/dsv41-fp4.md):
  pr         linear e4m3 scale tiles, B loaded as [K/2, N] pointers      (vLLM#38338, measured on a 5090)
  pr_perm    linear scales, B loaded [N, K/2] then tl.permute            (Helion's B)
  bytes      swizzled e4m3 byte gather, B permuted                       (body nvfp4_gemm_q_fp4mma_b)
  words      swizzled int32 word gather + join/unpack + bitcast          (body nvfp4_gemm_q_fp4mma)
  linwords   linear int32 words + join/unpack                            (body ..._fp4mma_lin)
  cvt        hardware cvt.rn.f16x2.e2m1x2 decode (inline asm) + f16 MMA  (body nvfp4_gemm_q_fp4cvt)
A variant that compiles but is not native is the finding "Triton silently decomposed this pattern".
Exit 0 = every checked variant of the primary set (pr, words) native on every config; 1 = not; 2 = no Triton.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SM120_SMEM = 101376
K_DEFAULT, N_DEFAULT = 5120, 4096

# ---------------------------------------------------------------------------
# the mirror kernel (source text: Triton's @jit needs a real file; written to a temp dir at run time)
# ---------------------------------------------------------------------------
MIRROR_SRC = textwrap.dedent('''
    import triton
    import triton.language as tl


    @triton.jit
    def mirror(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M,
               N: tl.constexpr, K: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               SCALES: tl.constexpr, B_PERM: tl.constexpr, QRED: tl.constexpr):
        # SCALES 0 linear e4m3 bytes | 1 swizzled int32 words | 2 swizzled e4m3 bytes | 3 linear int32 words
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        KH: tl.constexpr = K // 2
        offs_k = tl.arange(0, BLOCK_K // 2)
        a_ptrs = a_ptr + offs_m[:, None] * KH + offs_k[None, :]
        if B_PERM:
            b_ptrs = b_ptr + offs_n[:, None] * KH + offs_k[None, :]
        else:
            b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * KH
        NQ: tl.constexpr = BLOCK_K // 64
        NB: tl.constexpr = BLOCK_K // 16
        RW: tl.constexpr = (K // 64) * 128
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for it in tl.range(0, K // BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if B_PERM:
                b = tl.permute(b, (1, 0))
            if QRED:
                # what Helion emits: quad / block ids derived from the K-tile index vector by a reduce
                idx = it * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
                q = tl.min(tl.reshape(idx, (NQ, 32)), axis=1) >> 5
                kb = tl.min(tl.reshape(idx, (NB, 8)), axis=1) >> 3
            else:
                q = it * NQ + tl.arange(0, NQ)
                kb = it * NB + tl.arange(0, NB)
            if SCALES == 0:
                sa = tl.load(as_ptr + offs_m[:, None] * (K // 16) + kb[None, :])
                sb = tl.load(bs_ptr + offs_n[:, None] * (K // 16) + kb[None, :])
            elif SCALES == 2:
                ra = (offs_m >> 7) * (RW * 4) + (offs_m & 31) * 16 + ((offs_m >> 5) & 3) * 4
                rb = (offs_n >> 7) * (RW * 4) + (offs_n & 31) * 16 + ((offs_n >> 5) & 3) * 4
                sa = tl.load(as_ptr + ra[:, None] + (kb >> 2)[None, :] * 512 + (kb & 3)[None, :])
                sb = tl.load(bs_ptr + rb[:, None] + (kb >> 2)[None, :] * 512 + (kb & 3)[None, :])
            else:
                if SCALES == 1:
                    wa = (offs_m >> 7) * RW + (offs_m & 31) * 4 + ((offs_m >> 5) & 3)
                    wb = (offs_n >> 7) * RW + (offs_n & 31) * 4 + ((offs_n >> 5) & 3)
                    pa = tl.load(as_ptr + wa[:, None] + q[None, :] * 128)
                    pb = tl.load(bs_ptr + wb[:, None] + q[None, :] * 128)
                else:
                    pa = tl.load(as_ptr + offs_m[:, None] * (K // 64) + q[None, :])
                    pb = tl.load(bs_ptr + offs_n[:, None] * (K // 64) + q[None, :])
                sa = tl.join(tl.join(pa & 255, (pa >> 16) & 255), tl.join((pa >> 8) & 255, (pa >> 24) & 255))
                sa = tl.reshape(sa, (BLOCK_M, NB)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
                sb = tl.join(tl.join(pb & 255, (pb >> 16) & 255), tl.join((pb >> 8) & 255, (pb >> 24) & 255))
                sb = tl.reshape(sb, (BLOCK_N, NB)).to(tl.uint8).to(tl.float8e4nv, bitcast=True)
            acc = tl.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc, lhs_k_pack=True, rhs_k_pack=True)
            a_ptrs += BLOCK_K // 2
            b_ptrs += BLOCK_K // 2
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc.to(tl.bfloat16),
                 mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


    FP4X16_TO_F16_ASM = tl.constexpr("""
    {
        .reg .b32 lo, hi;
        .reg .b8 c0,c1,c2,c3,c4,c5,c6,c7;
        .reg .b32 h0,h1,h2,h3,h4,h5,h6,h7;
        mov.b64 {lo, hi}, $16;
        mov.b32 {c0,c1,c2,c3}, lo;
        cvt.rn.f16x2.e2m1x2 h0, c0; cvt.rn.f16x2.e2m1x2 h1, c1;
        cvt.rn.f16x2.e2m1x2 h2, c2; cvt.rn.f16x2.e2m1x2 h3, c3;
        mov.b32 {c4,c5,c6,c7}, hi;
        cvt.rn.f16x2.e2m1x2 h4, c4; cvt.rn.f16x2.e2m1x2 h5, c5;
        cvt.rn.f16x2.e2m1x2 h6, c6; cvt.rn.f16x2.e2m1x2 h7, c7;
        mov.b32 {$0,$1}, h0; mov.b32 {$2,$3}, h1;
        mov.b32 {$4,$5}, h2; mov.b32 {$6,$7}, h3;
        mov.b32 {$8,$9}, h4; mov.b32 {$10,$11}, h5;
        mov.b32 {$12,$13}, h6; mov.b32 {$14,$15}, h7;
    }
    """)


    @triton.jit
    def mirror_cvt(a_ptr, w_ptr, c_ptr, M, N: tl.constexpr, K: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # f16 activation [M, K] x hardware-decoded e2m1 weight [N, K/2] (the weight half of
        # nvfp4_gemm_q_fp4cvt): proves cvt.rn.f16x2.e2m1x2 assembles for sm_120a and the f16 MMA path
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        G: tl.constexpr = BLOCK_K // 16
        offs_g = tl.arange(0, G)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w64 = w_ptr.to(tl.pointer_type(tl.uint64))
        for it in tl.range(0, K // BLOCK_K):
            a = tl.load(a_ptr + offs_m[:, None] * K + it * BLOCK_K + offs_k[None, :])
            qw = tl.load(w64 + offs_n[:, None] * (K // 16) + it * G + offs_g[None, :])
            lanes = tl.inline_asm_elementwise(
                FP4X16_TO_F16_ASM, "=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,=h,l", [qw],
                (tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16,
                 tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16, tl.float16,
                 tl.float16, tl.float16), is_pure=True, pack=1)
            w = tl.join(tl.join(tl.join(tl.join(lanes[0], lanes[8]), tl.join(lanes[4], lanes[12])),
                                tl.join(tl.join(lanes[2], lanes[10]), tl.join(lanes[6], lanes[14]))),
                        tl.join(tl.join(tl.join(lanes[1], lanes[9]), tl.join(lanes[5], lanes[13])),
                                tl.join(tl.join(lanes[3], lanes[11]), tl.join(lanes[7], lanes[15]))))
            w = tl.reshape(w, (BLOCK_N, BLOCK_K))
            acc = tl.dot(a, tl.trans(w), acc)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc.to(tl.bfloat16),
                 mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))
''')

# (name, SCALES, B_PERM, QRED): the dot_scaled variants
VARIANTS = {
    "pr": (0, 0, 0),
    "pr_perm": (0, 1, 0),
    "bytes": (2, 1, 1),
    "words": (1, 1, 1),
    "linwords": (3, 1, 1),
}
PRIMARY = ("pr", "words")  # the exit code judges these


def configs(quick: bool) -> list:
    """(BM, BN, BK elements, warps, stages): the dossier T2 sweep (BM {16,32,64,128}, BN {32..256},
    BK {64,128,256}, warps {2,4,8}, stages {2..5}); quick = the decode-relevant corner + the
    vLLM#38338 winners."""
    if quick:
        return [(16, 64, 256, 4, 4), (16, 32, 256, 4, 4), (16, 64, 128, 4, 3), (16, 64, 256, 8, 3),
                (32, 64, 256, 4, 5), (32, 128, 128, 4, 3), (64, 64, 256, 4, 4), (64, 128, 256, 4, 4),
                (128, 128, 128, 8, 3), (16, 128, 64, 4, 2), (16, 256, 128, 8, 3), (64, 256, 256, 8, 3)]
    return [(bm, bn, bk, w, s) for bm, bn, bk, w, s in itertools.product(
        (16, 32, 64, 128), (32, 64, 128, 256), (64, 128, 256), (2, 4, 8), (2, 3, 4, 5))]


# ---------------------------------------------------------------------------
# pure helpers (testable without Triton)
# ---------------------------------------------------------------------------
def result_row(variant, cfg, info, stats, smem, wall_s, ptx_problems) -> dict:
    bm, bn, bk, warps, stages = cfg
    return {"variant": variant, "bm": bm, "bn": bn, "bk": bk, "warps": warps, "stages": stages,
            "native": bool(info["native_mma"]), "n_native_mma": info["native_mma"],
            "other_mma": info["other_mma"], "target": info["target"], "local": info["local"],
            "cp_async": info["cp_async"], "regs": stats.get("regs"), "smem_ptxas": stats.get("smem"),
            "smem_triton": smem, "spill_stores": stats.get("spill_stores"),
            "spill_loads": stats.get("spill_loads"), "stack": stats.get("stack"),
            "problems": ptx_problems + ([f"smem {smem} > {SM120_SMEM}"] if smem and smem > SM120_SMEM else []),
            "wall_s": round(wall_s, 1)}


def legal(row: dict) -> bool:
    """native, spill-free, fits 99 KB, and nothing else wrong."""
    return (not row.get("error") and row["native"] and not row["problems"]
            and not (row.get("spill_stores") or 0) and not (row.get("spill_loads") or 0))


def summarize(rows: list) -> dict:
    """Per variant: counts and the legal-tile table + the suggested register_block_size bounds
    (BM, BN, BK-in-BYTES = elements / 2) of the Helion bodies: the min / max of the tiles that
    were legal for at least one (warps, stages)."""
    out = {}
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v]
        ok = [r for r in rs if legal(r)]
        tiles = sorted({(r["bm"], r["bn"], r["bk"]) for r in ok})
        out[v] = {
            "compiled": sum(1 for r in rs if not r.get("error")),
            "errors": sum(1 for r in rs if r.get("error")),
            "native": sum(1 for r in rs if r.get("native")),
            "legal": len(ok), "total": len(rs),
            "decomposed": [(r["bm"], r["bn"], r["bk"], r["warps"], r["stages"]) for r in rs
                           if not r.get("error") and not r["native"]][:12],
            "legal_tiles": tiles,
            "suggested_bounds": ({"BM": [min(t[0] for t in tiles), max(t[0] for t in tiles)],
                                  "BN": [min(t[1] for t in tiles), max(t[1] for t in tiles)],
                                  "BK_bytes": [min(t[2] for t in tiles) // 2, max(t[2] for t in tiles) // 2]}
                                 if tiles else None),
        }
    return out


def compare(a: dict, b: dict) -> list:
    """Lines comparing two result files (3.7.1 vs 3.8.0): per variant, configs native in one and
    not in the other, and register / smem deltas on the common legal set."""
    key = lambda r: (r["variant"], r["bm"], r["bn"], r["bk"], r["warps"], r["stages"])
    ra, rb = {key(r): r for r in a["rows"]}, {key(r): r for r in b["rows"]}
    lines = [f"A = triton {a['triton']} (ptxas {a['ptxas']}), B = triton {b['triton']} (ptxas {b['ptxas']})"]
    for k in sorted(set(ra) & set(rb)):
        x, y = ra[k], rb[k]
        if x.get("error") or y.get("error") or legal(x) != legal(y):
            lines.append(f"{k}: legal A={legal(x)} B={legal(y)} A.problems={x.get('problems')} "
                         f"B.problems={y.get('problems')}")
        elif legal(x) and (x["regs"], x["smem_triton"]) != (y["regs"], y["smem_triton"]):
            lines.append(f"{k}: regs {x['regs']} -> {y['regs']}, smem {x['smem_triton']} -> {y['smem_triton']}")
    return lines


# ---------------------------------------------------------------------------
# Triton side (Linux, unrun by the author)
# ---------------------------------------------------------------------------
def load_mirror(tmpdir):
    import importlib.util
    path = os.path.join(tmpdir, "fp4mma_mirror.py")
    with open(path, "w") as f:
        f.write(MIRROR_SRC)
    spec = importlib.util.spec_from_file_location("fp4mma_mirror", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fp4mma_mirror"] = mod
    spec.loader.exec_module(mod)
    return mod


def ptxas_path(arch: int) -> str:
    from triton.backends.nvidia.compiler import get_ptxas
    return get_ptxas(arch).path


def ptxas_stats(ptx: str, arch: int, gpu_name: str = "sm_120a") -> dict:
    """`ptxas -v` on the PTX (registers, spill bytes, smem), the numbers Triton hides in a log."""
    from suffix_hybrid.kernels.fp4mma_probe import parse_ptxas_v
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "k.ptx"), os.path.join(d, "k.o")
        with open(src, "w") as f:
            f.write(ptx)
        proc = subprocess.run([ptxas_path(arch), "-v", f"--gpu-name={gpu_name}", src, "-o", dst],
                              capture_output=True, text=True)
        if proc.returncode:
            raise RuntimeError(f"ptxas failed ({proc.returncode}): {proc.stderr[-800:]}")
        return parse_ptxas_v(proc.stderr)


def compile_one(job: dict) -> dict:
    """Compile one (variant, config) for GPUTarget("cuda", arch, 32); runs in a worker process."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from suffix_hybrid.kernels.fp4mma_probe import analyze_ptx, ptx_problems

    variant, cfg, arch, k, n = job["variant"], tuple(job["cfg"]), job["arch"], job["k"], job["n"]
    bm, bn, bk, warps, stages = cfg
    t0 = time.time()
    mod = load_mirror(job["tmpdir"])
    div16 = lambda *ix: {(i,): [["tt.divisibility", 16]] for i in ix}
    try:
        if variant == "cvt":
            sig = {"a_ptr": "*fp16", "w_ptr": "*u8", "c_ptr": "*bf16", "M": "i32", "N": "constexpr",
                   "K": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr",
                   "BLOCK_K": "constexpr"}
            consts = {"N": n, "K": k, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}
            fn, attrs = mod.mirror_cvt, div16(0, 1, 2)
        else:
            scales, b_perm, qred = VARIANTS[variant]
            sty = "*fp8e4nv" if scales in (0, 2) else "*i32"
            sig = {"a_ptr": "*u8", "b_ptr": "*u8", "c_ptr": "*bf16", "as_ptr": sty, "bs_ptr": sty,
                   "M": "i32", "N": "constexpr", "K": "constexpr", "BLOCK_M": "constexpr",
                   "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "SCALES": "constexpr",
                   "B_PERM": "constexpr", "QRED": "constexpr"}
            consts = {"N": n, "K": k, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SCALES": scales,
                      "B_PERM": b_perm, "QRED": qred}
            fn, attrs = mod.mirror, div16(0, 1, 2, 3, 4)
        target = GPUTarget("cuda", arch, 32)
        src = ASTSource(fn=fn, signature=sig, constexprs=consts, attrs=attrs)
        ck = triton.compile(src, target=target, options={"num_warps": warps, "num_stages": stages})
        ptx = ck.asm["ptx"]
        info = analyze_ptx(ptx)
        why = ptx_problems(info, allow_local=False, want_native=variant != "cvt")
        if variant == "cvt" and "cvt.rn.f16x2.e2m1x2" not in ptx:
            why.append("no cvt.rn.f16x2.e2m1x2 in the PTX")
        row = result_row(variant, cfg, info, ptxas_stats(ptx, arch), getattr(ck.metadata, "shared", None),
                         time.time() - t0, why)
        if job.get("save_ptx"):
            with open(os.path.join(job["save_ptx"], f"{variant}_{bm}x{bn}x{bk}_w{warps}s{stages}.ptx"), "w") as f:
                f.write(ptx)
        return row
    except Exception as exc:  # noqa: BLE001 — a compile failure IS the finding for that config
        return {"variant": variant, "bm": bm, "bn": bn, "bk": bk, "warps": warps, "stages": stages,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}", "native": False, "problems": [],
                "wall_s": round(time.time() - t0, 1)}


def run_mirror(args) -> dict:
    import triton
    from concurrent.futures import ProcessPoolExecutor
    tv = triton.__version__
    from suffix_hybrid.kernels.fp4mma_probe import MIN_TRITON, parse_version
    if (parse_version(tv) or (0, 0, 0)) < MIN_TRITON:
        raise SystemExit(f"triton {tv} < {MIN_TRITON}: no native FP4 lowering for SM120 (triton#8494)")
    ptxas = ptxas_path(args.arch)
    ver = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print(f"triton {tv}, ptxas {ptxas} ({ver})", flush=True)
    names = list(VARIANTS) + (["cvt"] if args.cvt else [])
    names = [v for v in names if not args.variants or v in args.variants.split(",")]
    tmp = tempfile.mkdtemp(prefix="fp4mma_p0_")
    jobs = [{"variant": v, "cfg": list(c), "arch": args.arch, "k": args.k, "n": args.n, "tmpdir": tmp,
             "save_ptx": args.save_ptx} for v in names for c in configs(args.quick)
            if c[2] <= args.k and (v != "cvt" or c[2] % 16 == 0)]
    if args.save_ptx:
        os.makedirs(args.save_ptx, exist_ok=True)
    print(f"{len(jobs)} compiles ({len(names)} variants x {len(jobs) // max(1, len(names))} configs), "
          f"{args.jobs} jobs", flush=True)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(compile_one, jobs))
    else:
        rows = [compile_one(j) for j in jobs]
    return {"triton": tv, "ptxas": ver, "arch": args.arch, "k": args.k, "n": args.n, "rows": rows,
            "summary": summarize([r for r in rows if not r.get("error")] + [r for r in rows if r.get("error")])}


# ---------------------------------------------------------------------------
# --helion: the REAL bodies, pushed through JITFunction.run with a fake active driver
# ---------------------------------------------------------------------------
class FakeDriver:
    """Duck-typed triton driver: just enough for JITFunction.run / create_binder on a GPU-less box
    (triton/runtime/jit.py 3.7.1 / 3.8.0 use get_current_device, get_current_stream, get_current_target)."""

    def __init__(self, arch: int = 120):
        from triton.backends.compiler import GPUTarget
        self.target = GPUTarget("cuda", arch, 32)

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return self.target

    def map_python_to_cpp_type(self, ty):
        raise NotImplementedError("the fake driver never launches")


def moe_calls(fe, e: int, h: int, i: int, topk: int = 6, m: int = 16) -> dict:
    """Host-function arguments of the three MoE bodies (route_quant, fc1, fc2) at the given H / I and a
    small expert count E (E only changes the grid and constants, not the instruction mix or the register
    use); lane B's combine is not ours to probe."""
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    p = nm.make_problem(e, h, i, "silu", "cpu", 0, limit=10.0)
    ids, tw = nm.rand_routing(m, 2 * e, topk, "cpu", 1, 2, e)
    pairs = m * topk
    counts = torch.zeros(e, dtype=torch.int32)
    ps = torch.zeros(pairs + 1, dtype=torch.int32)
    sf = lambda rows, cols: fe.sf_arg(torch.zeros(rows, cols, dtype=torch.uint8))  # in fe.SCALE_READ's form
    return {
        fe.ROUTE_QUANT: (torch.randn(m, h).bfloat16(), ids, 1.0, 1.0, e, 2),
        fe.FC1: (torch.zeros(m, h // 2, dtype=torch.uint8), sf(m, h // 16), ps,
                 counts, counts, p["w13"], fe.sf_arg(p["w13_sf"]), p["g1"], 1.0, 1.0, 0, topk, 10.0),
        fe.FC2: (torch.zeros(pairs + 1, i // 2, dtype=torch.uint8), sf(pairs + 1, i // 16), ps, counts, counts,
                 tw, p["w2"], fe.sf_arg(p["w2_sf"]), p["g2"]),
    }


def helion_bodies(args) -> list:
    """(label, host function name, source text, cpu args, want_native) for the real bodies: the dense GEMM
    bodies at (N, K) = (--n, --k), M 16, and the three MoE bodies at --moe-h / --moe-i / --moe-e (the V4.1
    EP8 rank's H and I, a handful of experts), each at an explicit realistic config. Needs Helion and
    torch only (the Helion front end lowers on CPU); the compile happens in run_helion."""
    import helion
    from helion.runtime.ref_mode import RefMode
    fams = {s.strip() for s in args.helion_families.split(",") if s.strip()}
    out = []
    if "gemm" in fams:
        from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
        n, k, m = (args.n, args.k, 16)
        cfg = dict(block_sizes=[16, 64, 128], num_warps=4, num_stages=3, split_k=1)
        for kind in (fm.WORDS, fm.BYTES, fm.LIN, fm.CVT):
            p = fm.problem({"k": k, "n": n, "numtokens": m}, 1, "cpu", kind)
            a = fm.body_args(kind, p)
            kern = helion.kernel(fm.BODIES[kind], ref_mode=RefMode.OFF, static_shapes=False,
                                 ignore_warnings=fm._ignore())
            out.append((f"gemm.{kind}", kind, kern.bind(a).to_triton_code(helion.Config(**cfg)), a,
                        kind != fm.CVT))
    if "moe" in fams:
        from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fe
        mode0 = fe.SCALE_READ
        try:
            # fc1 / fc2 in BOTH scale-read modes (words = join/reshape/bitcast unpack, bytes = plain e4m3
            # tile loads: the fallback P0 may need); the other three bodies do not read scales
            for mode in ("words", "bytes"):
                fe.SCALE_READ = mode
                calls = moe_calls(fe, args.moe_e, args.moe_h, args.moe_i)
                for name, a in calls.items():
                    if mode == "bytes" and name not in fe.NATIVE_KERNELS:
                        continue
                    kern = helion.kernel(fe.BODIES[name], ref_mode=RefMode.OFF, static_shapes=False,
                                         ignore_warnings=[helion.exc.TensorOperationInWrapper])
                    code = kern.bind(a).to_triton_code(helion.Config(**fe.PROBE_CFG[name]))
                    label = f"moe.{name}" + ("[bytes]" if mode == "bytes" else "")
                    out.append((label, name, code, a, name in fe.NATIVE_KERNELS))
        finally:
            fe.SCALE_READ = mode0
    return out


def run_helion(args) -> list:
    """Compile the real generated kernels for sm_120a. For each body: write Helion's generated
    module to a file, import it (so @triton.jit can read its source), install the fake driver and a
    jit_cache_hook that compiles with ASTSource for GPUTarget("cuda", 120, 32), keeps the PTX and
    returns True (no launch), call the generated host function with CPU tensors, restore."""
    import importlib.util
    import triton
    import triton.runtime.driver as drv
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from suffix_hybrid.kernels.fp4mma_probe import analyze_ptx, ptx_problems

    tmp = tempfile.mkdtemp(prefix="fp4mma_p0_helion_")
    fake = FakeDriver(args.arch)
    target = GPUTarget("cuda", args.arch, 32)
    rows = []
    for label, kind, code, call_args, want_native in helion_bodies(args):
        modname = "gen_" + re.sub(r"\W", "_", label)  # unique per label (fc1 exists in both scale-read modes)
        path = os.path.join(tmp, f"{modname}.py")
        with open(path, "w") as f:
            f.write(code)
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        host = getattr(mod, kind)
        got = {}

        def hook(*, key, repr, fn, compile, is_manual_warmup, already_compiled):  # noqa: A002
            src = ASTSource(fn.jit_function, compile["signature"], compile["constants"],
                            compile["configs"][0])
            opts = {"num_warps": compile["num_warps"], "num_stages": compile["num_stages"],
                    "num_ctas": compile["num_ctas"], "enable_fp_fusion": compile["enable_fp_fusion"]}
            got["ck"] = triton.compile(src, target=target, options=opts)
            return True

        old_driver, old_hook = drv.driver._active, knobs.runtime.jit_cache_hook
        drv.driver.set_active(fake)
        knobs.runtime.jit_cache_hook = hook
        t0 = time.time()
        try:
            host(*call_args)
            ck = got["ck"]
            info = analyze_ptx(ck.asm["ptx"])
            why = ptx_problems(info, allow_local=False, want_native=want_native)
            rows.append(result_row(label, (0, 0, 0, ck.metadata.num_warps, ck.metadata.num_stages), info,
                                   ptxas_stats(ck.asm["ptx"], args.arch), ck.metadata.shared,
                                   time.time() - t0, why))
        except Exception as exc:  # noqa: BLE001
            rows.append({"variant": label, "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                         "native": False, "problems": [], "bm": 0, "bn": 0, "bk": 0, "warps": 0,
                         "stages": 0})
        finally:
            drv.driver._active = old_driver
            knobs.runtime.jit_cache_hook = old_hook
    return rows


# ---------------------------------------------------------------------------
def selftest() -> int:
    """No Triton needed: the mirror source parses, the sweeps have the advertised size, summarize /
    compare / the PTX judge behave on canned rows."""
    import ast
    ast.parse(MIRROR_SRC)
    assert len(configs(False)) == 4 * 4 * 3 * 3 * 4 and len(configs(True)) == 12
    from suffix_hybrid.kernels.fp4mma_probe import NATIVE_MMA, analyze_ptx
    ptx = (".version 9.1\n.target sm_120a\n.visible .entry k(\n)\n{\n  " + NATIVE_MMA +
           " {%f1,%f2,%f3,%f4}, {%r1,%r2,%r3,%r4}, {%r5,%r6}, {%f1,%f2,%f3,%f4}, {%r7}, {0,0}, {%r8}, {0,0};\n}\n")
    info = analyze_ptx(ptx)
    row = result_row("words", (16, 64, 256, 4, 4), info, {"regs": 90, "smem": 20480, "spill_stores": 0,
                                                          "spill_loads": 0}, 20480, 1.0, [])
    assert legal(row)
    bad = dict(row, native=False, problems=["decomposed"])
    s = summarize([row, bad, dict(row, bm=64, bn=128, bk=256)])
    assert s["words"]["legal"] == 2 and s["words"]["suggested_bounds"]["BM"] == [16, 64]
    assert s["words"]["suggested_bounds"]["BK_bytes"] == [128, 128]
    a = {"triton": "3.7.1", "ptxas": "x", "rows": [row]}
    b = {"triton": "3.8.0", "ptxas": "y", "rows": [dict(row, regs=120)]}
    assert any("regs 90 -> 120" in ln for ln in compare(a, b)), compare(a, b)
    print("fp4mma_p0_probe selftest OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="12 decode-relevant configs per variant")
    ap.add_argument("--variants", default="", help="comma list of pr,pr_perm,bytes,words,linwords")
    ap.add_argument("--cvt", action="store_true", help="also compile the hardware-cvt mirror")
    ap.add_argument("--helion", action="store_true", help="also compile the real Helion bodies (fake driver)")
    ap.add_argument("--helion-families", default="gemm,moe",
                    help="which real bodies --helion compiles: gemm (dense NVFP4 GEMM), moe (V4.1 routed experts)")
    ap.add_argument("--moe-e", type=int, default=4, help="experts of the --helion MoE layer (grid only)")
    ap.add_argument("--moe-h", type=int, default=5120, help="hidden size of the --helion MoE layer (V4.1: 5120)")
    ap.add_argument("--moe-i", type=int, default=2304, help="moe intermediate size (V4.1 EP8: 2304)")
    ap.add_argument("--arch", type=int, default=120)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default="")
    ap.add_argument("--save-ptx", default="", help="directory for the PTX of every compile")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        print("\n".join(compare(a, b)))
        return 0
    try:
        import triton  # noqa: F401
    except ImportError:
        print("triton is not installed (Linux wheels only): nothing to probe", file=sys.stderr)
        return 2
    res = run_mirror(args)
    if args.helion:
        res["helion_rows"] = run_helion(args)
    print(json.dumps(res["summary"], indent=1, default=list))
    for r in res.get("helion_rows", []):
        print("helion", json.dumps(r, default=list))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, default=list)
    prim = [r for r in res["rows"] if r["variant"] in PRIMARY]
    bad = [r for r in prim if r.get("error") or not r["native"]]
    print(f"primary variants {PRIMARY}: {len(prim) - len(bad)}/{len(prim)} native "
          f"({'OK' if not bad else 'NOT ALL NATIVE: ' + str([(r['variant'], r['bm'], r['bn'], r['bk'], r.get('error')) for r in bad[:6]])})")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
