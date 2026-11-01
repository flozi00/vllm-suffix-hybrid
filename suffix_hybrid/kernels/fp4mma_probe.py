# SPDX-License-Identifier: Apache-2.0
"""K5 plumbing, P0 helpers and the boot gate of the native-FP4-MMA family
(helion_kernels/nvfp4_gemm_fp4mma.py, helion_kernels/dsv41_moe_fp4mma.py; lane I).

1. STACK GATE. `require_native_stack()` is the startup error with the reason: the GPU is not
   SM12x (family 120 is checked explicitly, never family 100), Triton < 3.6.0 (the native
   `tl.dot_scaled` lowering, triton#8494, is in v3.6.0), Helion missing.

2. K5, the PTX assertion. Triton can SILENTLY decompose `tl.dot_scaled` into a bf16 upcast + a
   regular MMA when its `ScaledBlockedToMMA` pattern does not match (a Linear-layout scale operand,
   K < 64 per tile, a mixed-precision pair, ...): the kernel is correct and several times slower, and
   nothing says so. So every kernel of the family is checked at lane warm-up: its compiled PTX must
   contain `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.
   e2m1.f32.ue4m3` for `.target sm_120a`, else `Fp4mmaError` (fail closed, no fallback).
   PLUMBING (read from helion 1.5.x / triton 3.7.1 / 3.8.0 source): Helion's generated host code
   calls `_launcher(triton_kernel, grid, *args, num_warps=.., num_stages=..)` and DROPS the return
   value; `helion.runtime.default_launcher` (bound into every generated module at its first import
   as `_default_launcher`) returns `triton_kernel.run(...)` = the CompiledKernel, with `.asm["ptx"]`,
   `.n_regs`, `.n_spills` (after the first launch loaded it) and `.metadata.shared`. There is no
   official hook, so `install_recorder()` swaps `helion.runtime.default_launcher` for a recording
   wrapper BEFORE the family's first kernel is compiled (the dense `prepare()` does it; the MoE
   `run_plan` does it on its first GPU call and then judges every newly compiled fc1 / fc2 right after
   the launch, so lane B's seam needs no K5 line of its own); kernels compiled earlier (other lanes)
   keep the original launcher and are never recorded, and `install_recorder(watch=...)` keeps only
   the family's own kernels (exact Triton names, `name_matches`). If the swap cannot be made
   (a Helion without that attribute) the assertion cannot run and the lane refuses to start; the
   fallback for such a build is Triton's own cache (TRITON_CACHE_DIR=<fresh dir> + `ptx_from_cache`).

3. P0 helpers (scripts/fp4mma_p0_probe.py): `analyze_ptx`, `ptx_problems`, `parse_ptxas_v`.

4. THE BOOT GATE `fp4mma_capture_replay` (`python -m suffix_hybrid.kernels.fp4mma_probe
   capture_replay`; sitecustomize `_BOOT_GATES` entry in BOOT_GATES below): the capture-safety proof,
   in the style of the mxfp8 / moe_shared gates. Per probe (body x real shape x M x explicit config; plus
   every SHIPPED config of this platform): eager on three inputs with the output and every partial
   NaN-poisoned first (K3 vs the exact f64 product), capture the same call on
   a side stream, replay with NEW data bit-exact against eager, a guarded output (sentinel bands),
   and two graphs replayed concurrently on two streams; at the end `gate_k5` judges the compiled PTX of
   every body the probes ran (lab strictness: spills fail). Exit 0 PASS / 1 FAIL / 2 NOT RUN.
   compute-sanitizer works (the gate times and tunes nothing).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

MARKER = "[suffix fp4mma]"
MIN_TRITON = (3, 6, 0)  # triton#8494 "Add native FP4 scaled_dot for SM120", v3.6.0
NATIVE_MMA = ("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X."
              "f32.e2m1.e2m1.f32.ue4m3")
SM120_SMEM = 101376  # 99 KB per block on SM120 (dynamic shared memory opt-in ceiling)


class Fp4mmaError(RuntimeError):
    """A native-FP4-MMA contract was violated (stack, PTX, capture): always a startup error."""


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


logline = _log  # for the family modules (K5 arm lines)


# ---------------------------------------------------------------------------
# 1. stack gate
# ---------------------------------------------------------------------------
def parse_version(text: str | None):
    """'3.7.1' / '3.8.0+git1234' / '3.6.0rc1' -> (3, 7, 1); None when unparsable."""
    m = re.match(r"\s*(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    return None if m is None else (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def stack_problems(cc, triton_version, helion_ok: bool, knob: str = "") -> list:
    """Reasons the native FP4 MMA arm cannot run here (empty = it can). Pure: the tests pin each
    branch. cc = (major, minor) of the device or None (no CUDA)."""
    why = []
    if cc is None:
        why.append("no CUDA device (the arm is sm_120a SASS compiled by Triton)")
    elif cc[0] != 12:
        why.append(f"the GPU is cc {cc[0]}.{cc[1]}: native FP4 MMA (kind::mxf4nvf4) on this arm is "
                   "SM12x only (family 120; family 100 uses tcgen05, not these kernels)")
    tv = parse_version(triton_version)
    if tv is None:
        why.append("Triton is not installed or its version is unparsable "
                   f"({triton_version!r})")
    elif tv < MIN_TRITON:
        why.append(f"Triton {triton_version} < {'.'.join(map(str, MIN_TRITON))}: tl.dot_scaled has "
                   "no native FP4 lowering for SM120 before triton#8494 (v3.6.0)")
    if not helion_ok:
        why.append("Helion is not importable (hl.dot_scaled)")
    return why


def require_native_stack() -> None:
    """Raise Fp4mmaError naming every reason the native arm cannot run on this process."""
    import importlib.util
    cc = triton_v = None
    try:
        import torch
        if torch.cuda.is_available():
            cc = tuple(torch.cuda.get_device_capability())
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton
        triton_v = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    helion_ok = importlib.util.find_spec("helion") is not None
    why = stack_problems(cc, triton_v, helion_ok)
    if why:
        raise Fp4mmaError(f"{MARKER} the native FP4 MMA arm (SUFFIX_HELION_NVFP4_GEMM_FP4MMA=1, or a "
                          f"dsv41_moe_variant row selecting dot_scaled) cannot run: " + "; ".join(why))


# ---------------------------------------------------------------------------
# 2. PTX analysis (pure strings: the tests feed canned PTX, the P0 probe feeds real PTX)
# ---------------------------------------------------------------------------
def analyze_ptx(ptx: str) -> dict:
    """What the PTX of one kernel says about the tensor-core path."""
    target = re.search(r"^\s*\.target\s+(\S+)", ptx, re.M)
    mmas = re.findall(r"\bmma\.sync\.aligned\.[A-Za-z0-9_.:]+", ptx)
    native = [m for m in mmas if m == NATIVE_MMA]
    other = Counter(m for m in mmas if m != NATIVE_MMA)
    return {
        "target": target.group(1) if target else None,
        "native_mma": len(native),
        "other_mma": dict(other),
        "kind_mxf4nvf4": "kind::mxf4nvf4" in ptx,
        "scale_vec_4x": "block_scale.scale_vec::4X" in ptx,
        "ue4m3": "ue4m3" in ptx,
        "ue8m0_scales": ".ue8m0" in ptx,
        "local": bool(re.search(r"^\s*\.local\b", ptx, re.M)),
        "cp_async": ptx.count("cp.async"),
        "ldmatrix": ptx.count("ldmatrix"),
        "bf16_mma": sum(v for k, v in other.items() if ".bf16.bf16" in k),
        "lines": ptx.count("\n"),
    }


def ptx_problems(info: dict, *, allow_local: bool = True, want_native: bool = True) -> list:
    """Why a kernel is NOT the intended native FP4 MMA path (empty = it is). `allow_local`:
    serving only warns about local memory (a spilling config is slow, not wrong); the P0 probe
    and the lab gate pass allow_local=False (K2: no spills on the hot entry)."""
    why = []
    if info["target"] is None or not str(info["target"]).startswith("sm_120"):
        why.append(f"target {info['target']!r}, want sm_120a (family 120)")
    elif info["target"] not in ("sm_120a", "sm_120f"):
        why.append(f"target {info['target']!r} is not the architecture-specific sm_120a: the "
                   "block-scaled MMA is not available")
    if want_native and not info["native_mma"]:
        det = ", ".join(f"{k} x{v}" for k, v in sorted(info["other_mma"].items())) or "no mma at all"
        why.append(f"no `{NATIVE_MMA}` in the PTX ({det}): Triton decomposed tl.dot_scaled "
                   "(a bf16 upcast + regular MMA) — correct but several times slower; "
                   "K5 refuses it")
    if want_native and info["ue8m0_scales"] and not info["ue4m3"]:
        why.append("MXFP4 (ue8m0, scale_vec::2X) instruction instead of NVFP4 (ue4m3, 4X): the "
                   "scale operand is not float8e4nv")
    if info["local"] and not allow_local:
        why.append("local memory (.local): register spills or a stack frame on the hot entry")
    return why


_PTXAS_RES = (("regs", r"Used\s+(\d+)\s+registers"), ("smem", r"(\d+)\s+bytes\s+smem"),
              ("spill_stores", r"(\d+)\s+bytes\s+spill\s+stores"),
              ("spill_loads", r"(\d+)\s+bytes\s+spill\s+loads"),
              ("stack", r"(\d+)\s+bytes\s+stack\s+frame"))


def parse_ptxas_v(text: str) -> dict:
    """`ptxas -v` stderr -> {regs, smem, spill_stores, spill_loads, stack} (ints, 0 when the
    line has no such field: ptxas omits smem for kernels that use none). Takes the LAST entry
    function when a file has several (a Triton PTX has one)."""
    out = {}
    for key, pat in _PTXAS_RES:
        hits = re.findall(pat, text)
        out[key] = int(hits[-1]) if hits else 0
    return out


# ---------------------------------------------------------------------------
# 3. K5: the recorder
# ---------------------------------------------------------------------------
class Record(NamedTuple):
    name: str  # Triton function name, e.g. _helion_nvfp4_gemm_q_fp4mma
    key: str  # CompiledKernel hash
    ptx: str
    n_regs: int | None
    n_spills: int | None
    shared: int | None
    num_warps: int | None
    num_stages: int | None


_RECORDS: dict = {}
_CHECKED: set = set()  # record keys assert_native(only_new=True) already judged
_WATCH: tuple | None = None  # Triton function names record_kernel keeps; None = every kernel
_WATCHED: dict = {}  # memo: Triton function name -> kept? (the recorder sits on EVERY Helion launch: one dict get)


def name_matches(name: str, want: str) -> bool:
    """A Triton function name is `want` or Helion's numbered variant of it (`want_1`): NOT a longer name
    that merely starts with it (`_helion_dsv41_moe_fc1_sh` is lane B's shared-expert kernel, not fc1)."""
    return name == want or (name.startswith(want + "_") and name[len(want) + 1:].isdigit())


def install_recorder(watch=None) -> bool:
    """Swap helion.runtime.default_launcher for a wrapper that records the CompiledKernel the
    real launcher returns. Idempotent. MUST run before the family's first kernel compiles: a
    generated module binds `default_launcher` at its first import. `watch`: Triton function names to
    keep (accumulates over calls; default = record every kernel, which a serving process does not
    want: PTX of unrelated Helion kernels would pile up). Returns True when installed (or already),
    False when this Helion has no such attribute (K5 then cannot run)."""
    global _WATCH
    if watch is not None:
        _WATCH = tuple(dict.fromkeys((*(_WATCH or ()), *watch)))
        _WATCHED.clear()
    try:
        import helion.runtime as hrt
    except Exception:  # noqa: BLE001
        return False
    real = getattr(hrt, "default_launcher", None)
    if real is None:
        return False
    if getattr(real, "_fp4mma_recorder", False):
        return True

    def recording_launcher(triton_kernel, grid, *args, **kwargs):
        ck = real(triton_kernel, grid, *args, **kwargs)
        try:
            record_kernel(triton_kernel, ck)
        except Exception as exc:  # noqa: BLE001 — recording must never break a launch
            _log(f"recorder: {type(exc).__name__}: {exc}")
        return ck

    recording_launcher._fp4mma_recorder = True
    recording_launcher._real = real
    hrt.default_launcher = recording_launcher
    return True


def uninstall_recorder() -> None:
    global _WATCH
    _WATCH = None
    _WATCHED.clear()
    try:
        import helion.runtime as hrt
    except Exception:  # noqa: BLE001
        return
    cur = getattr(hrt, "default_launcher", None)
    if getattr(cur, "_fp4mma_recorder", False):
        hrt.default_launcher = cur._real


def recorder_installed() -> bool:
    try:
        import helion.runtime as hrt
        return bool(getattr(hrt.default_launcher, "_fp4mma_recorder", False))
    except Exception:  # noqa: BLE001
        return False


def record_kernel(triton_kernel, ck) -> None:
    """Keep the PTX / resource facts of a CompiledKernel (None = a warm-up/hook run: nothing)."""
    if ck is None or not hasattr(ck, "asm"):
        return
    name = getattr(getattr(triton_kernel, "fn", None), "__name__", None) or getattr(ck, "name", "?")
    kept = _WATCHED.get(name)
    if kept is None:
        kept = _WATCHED[name] = _WATCH is None or any(name_matches(name, w) for w in _WATCH)
    if not kept:
        return
    key = str(getattr(ck, "hash", id(ck)))
    if key in _RECORDS:
        return
    md = getattr(ck, "metadata", None)
    _RECORDS[key] = Record(
        name=name, key=key, ptx=str(ck.asm["ptx"]), n_regs=getattr(ck, "n_regs", None),
        n_spills=getattr(ck, "n_spills", None), shared=getattr(md, "shared", None),
        num_warps=getattr(md, "num_warps", None), num_stages=getattr(md, "num_stages", None))


def records(prefix: str = "") -> list:
    return [r for r in _RECORDS.values() if r.name.startswith(prefix)]


def record_count() -> int:
    return len(_RECORDS)


def clear_records() -> None:
    _RECORDS.clear()
    _CHECKED.clear()


def record_line(r: Record, info: dict | None = None) -> str:
    info = info or analyze_ptx(r.ptx)
    return (f"{r.name}[warps={r.num_warps} stages={r.num_stages}] regs={r.n_regs} "
            f"spills={r.n_spills} smem={r.shared} target={info['target']} "
            f"native_mma={info['native_mma']} cp.async={info['cp_async']} local={info['local']}")


def assert_native(names, *, allow_local: bool = True, log=_log, only_new: bool = False,
                  require: bool = True) -> list:
    """K5. `names` = Triton function names (a trailing `_<n>` of Helion's numbering is the same
    kernel; a longer name is not) the family served at warm-up. Each must have >= 1 recorded
    compile (`require`) and EVERY recorded compile must pass `ptx_problems`; else Fp4mmaError naming
    the kernel and the reason. Logs one line per compiled specialization. `only_new`: skip (and do
    not log again) the compiles an earlier call already judged OK: the serving check after every
    launch. Returns the checked records."""
    if not recorder_installed():
        raise Fp4mmaError(f"{MARKER} K5: the launcher recorder is not installed (call "
                          "install_recorder() before the first compile; this Helion may lack "
                          "helion.runtime.default_launcher) — refusing to serve without the PTX "
                          "assertion")
    checked = []
    for name in names:
        recs = [r for r in _RECORDS.values() if name_matches(r.name, name)]
        if not recs and require:
            raise Fp4mmaError(f"{MARKER} K5: no compiled kernel recorded for {name!r} — the lane "
                              "did not warm it up under the recorder, the planned kernel never ran, "
                              "or the PTX was not available (TRITON_STORE_BINARY_ONLY=1 drops it "
                              "from Triton's cache; unset it for these pools)")
        for r in recs:
            if only_new and r.key in _CHECKED:
                continue
            info = analyze_ptx(r.ptx)
            why = ptx_problems(info, allow_local=allow_local)
            log(("K5 ok  " if not why else "K5 FAIL ") + record_line(r, info))
            if why:
                raise Fp4mmaError(f"{MARKER} K5: {r.name} (warps {r.num_warps}, stages "
                                  f"{r.num_stages}): " + "; ".join(why))
            if info["local"] or (r.n_spills or 0) > 0:
                log(f"K5 WARN {r.name}: local memory / spills "
                    f"({r.n_spills} B spilled): slow, not wrong; retune the config")
            _CHECKED.add(r.key)
            checked.append(r)
    return checked


def assert_contains(names, tokens, what: str, log=_log) -> list:
    """K5 for a control kernel that is NOT the native MMA (the hardware-cvt arm): every recorded compile
    of each name must contain every token (e.g. `cvt.rn.f16x2.e2m1x2`), for sm_120a."""
    if not recorder_installed():
        raise Fp4mmaError(f"{MARKER} K5: the launcher recorder is not installed")
    checked = []
    for name in names:
        recs = [r for r in _RECORDS.values() if name_matches(r.name, name)]
        if not recs:
            raise Fp4mmaError(f"{MARKER} K5: no compiled kernel recorded for {name!r}")
        for r in recs:
            info = analyze_ptx(r.ptx)
            missing = [t for t in tokens if t not in r.ptx]
            if missing or not str(info["target"]).startswith("sm_120"):
                raise Fp4mmaError(f"{MARKER} K5: {r.name}: {what}: missing {missing}, target "
                                  f"{info['target']!r}")
            log("K5 ok  " + record_line(r, info))
            checked.append(r)
    return checked


def ptx_from_cache(cache_dir: str, name_prefix: str = "") -> dict:
    """Fallback when the launcher swap is impossible: {kernel name: ptx} from a FRESH Triton
    cache dir (TRITON_CACHE_DIR set before the first compile), whose <name>.ptx files all belong
    to this process. Triton writes one <hash>/<name>.ptx per compile."""
    out = {}
    for root, _dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".ptx") and f.startswith(name_prefix):
                with open(os.path.join(root, f)) as fh:
                    out[os.path.join(root, f)] = fh.read()
    return out


# ---------------------------------------------------------------------------
# 4. the capture/replay gate (probes live in the family modules; the CPU twin is EagerRuntime)
# ---------------------------------------------------------------------------
class CudaRuntime:
    """torch.cuda graph capture on a side stream."""
    name = "cuda"

    @staticmethod
    def sync() -> None:
        import torch
        torch.cuda.synchronize()

    @staticmethod
    def capture(fn):
        """fn() -> tensor; returns (that tensor at its graph address, replay)."""
        import torch
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            y = fn()
        return y, graph.replay


class EagerRuntime:
    """CPU twin for tests: 'capture' runs fn once, 'replay' re-runs it into the same tensor."""
    name = "eager"

    @staticmethod
    def sync() -> None:
        pass

    @staticmethod
    def capture(fn):
        y = fn()

        def replay():
            y.copy_(fn())
        return y, replay


def bits_equal(a, b) -> bool:
    import torch
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16),
                                              b.contiguous().view(torch.int16))


def capture_check(run, xs: list, copy_in, rt=None, replays: int | None = None, what: str = "") -> int:
    """`run(static) -> bf16 tensor` is the lane's work over a static input `static` (any object);
    `copy_in(static, x)` copies input `x` into it. Eager result per input (the first call
    warms / compiles outside any capture); capture run(static) on a side stream; replay `replays`
    times (default one per input, cycling) with NEW data and compare each replay bit-exactly with
    its eager result. Raises Fp4mmaError at the first mismatch. Returns the replays checked."""
    rt = rt or CudaRuntime
    first = xs[0]
    static = _clone_input(first)
    eager = []
    for x in xs:
        copy_in(static, x)
        eager.append(run(static).clone())
    rt.sync()
    copy_in(static, first)
    y, replay = rt.capture(lambda: run(static))
    rt.sync()
    count = replays or len(xs)
    for i in range(count):
        j = (i + 1) % len(xs)  # starts on NEW data (the capture saw input 0), then cycles
        copy_in(static, xs[j])
        replay()
        rt.sync()
        if not bits_equal(y, eager[j]):
            import torch
            bad = int((y.contiguous().view(torch.int16) != eager[j].contiguous().view(torch.int16)).sum())
            raise Fp4mmaError(
                f"{MARKER} CAPTURE/REPLAY FAIL {what}: replay {i} (input {j}) differs from the "
                f"eager result in {bad} of {y.numel()} elements")
    return count


def _clone_input(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_clone_input(e) for e in x)
    if isinstance(x, dict):
        return {k: _clone_input(v) for k, v in x.items()}
    return x


GUARD = 4096  # sentinel elements on each side of a guarded output
SENT = 0x5A5A  # int16 sentinel bit pattern (a finite bf16, not what a kernel writes)


def guarded(m: int, cols: int, dev):
    import torch
    arena = torch.full((GUARD + m * cols + GUARD,), SENT, dtype=torch.int16, device=dev)
    return arena, arena[GUARD:GUARD + m * cols].view(torch.bfloat16).view(m, cols)


def guards_intact(arena) -> bool:
    return bool((arena[:GUARD] == SENT).all()) and bool((arena[-GUARD:] == SENT).all())


@dataclass
class Probe:
    """One capture/replay probe (a pure description: the family module builds the Case)."""
    label: str
    family: str  # "gemm" | "moe"
    spec: dict = field(default_factory=dict)
    source: str = "probe"  # probe | shipped


@dataclass
class Case:
    """What `run_probe` needs from a family: `inputs` (a list of >= 3 distinct input objects),
    `run(static, out=None) -> bf16 tensor` (the lane's work over the static input; `out` = a
    caller-provided output view), `copy_in(static, x)` (x into the static input, in place),
    `exact(x, out) -> (ok, message)` (K3 against the exact f64 reference of input x) and
    `out_shape` = (rows, cols) of the output (for the guarded variant)."""
    inputs: list
    run: object
    copy_in: object
    exact: object
    out_shape: tuple


# sitecustomize runs a gate in a child that gets the pod env WITHOUT its SUFFIX_* keys plus the dict below, so
# the MoE scale-read mode the pool runs has to be a gate of its own (the dense bodies have no such mode).
# No enable knob is needed: the gate builds the kernels itself, and the MoE variant is selected by data.
BOOT_GATES = {
    "fp4mma_capture_replay": (["-m", "suffix_hybrid.kernels.fp4mma_probe", "capture_replay"], {}),
    "fp4mma_capture_replay_bytes": (["-m", "suffix_hybrid.kernels.fp4mma_probe", "capture_replay"],
                                    {"SUFFIX_FP4MMA_MOE_SCALE_READ": "bytes"}),
}


def all_probes(tier: str) -> list:
    """Dense GEMM probes (helion_kernels/nvfp4_gemm_fp4mma.probes) + MoE probes
    (helion_kernels/dsv41_moe_fp4mma.probes, when that module is present)."""
    out = []
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    out += fm.probes(tier)
    try:
        from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fmoe
    except ImportError:
        fmoe = None
    if fmoe is not None:
        out += fmoe.probes(tier)
    return out


def run_probe(pr: Probe, build, rt, dev, replays: int, streams: int, capture: bool,
              guards: bool) -> dict:
    """The gate for one probe; returns the record dict (status PASS | FAIL | NOT RUN)."""
    rec = {"label": pr.label, "family": pr.family, "source": pr.source, **pr.spec}
    try:  # --- 1. build + eager, compile outside any capture (a failure here is NOT RUN, not FAIL)
        case = build(pr, dev)
        static = _clone_input(case.inputs[0])
        case.copy_in(static, case.inputs[0])
        eager0 = case.run(static).clone()
        rt.sync()
    except Exception as exc:  # noqa: BLE001
        # a K5 refusal (PTX without the native MMA) raised by the family's own run_plan is the finding
        rec.update(status="FAIL" if isinstance(exc, Fp4mmaError) else "NOT RUN",
                   detail=f"{type(exc).__name__}: {exc}"[:600])
        return rec
    try:
        ok, msg = case.exact(case.inputs[0], eager0)
        if not ok:
            rec.update(status="FAIL", detail=f"eager vs exact: {msg}")
            return rec
        rec["check"] = msg
        if not capture:
            rec.update(status="PASS", detail="eager only (--no-capture)")
            return rec
        n_ok = capture_check(case.run, case.inputs, case.copy_in, rt, replays, what=pr.label)  # 2
        if guards:  # --- 3
            arena, view = guarded(*case.out_shape, dev)

            def run_g(static):
                view.fill_(float("nan"))
                return case.run(static, out=view)
            capture_check(run_g, case.inputs, case.copy_in, rt, replays, what=pr.label + " [guarded out]")
            rt.sync()
            if not guards_intact(arena):
                raise Fp4mmaError(f"{MARKER} GUARD FAIL {pr.label}: a store landed outside `out`")
        concurrent = streams >= 2 and rt is CudaRuntime
        if concurrent:  # --- 4
            _concurrent(case, dev, max(2, replays))
        rec.update(status="PASS", detail=f"{n_ok} replays bit-exact" + (", guards intact" if guards else "")
                   + (", 2-stream concurrent ok" if concurrent else ""))
    except Exception as exc:  # noqa: BLE001 — a capture / replay / guard failure is the finding
        rec.update(status="FAIL", detail=f"{type(exc).__name__}: {exc}"[:900])
    return rec


def _concurrent(case, dev, rounds: int) -> None:
    """Two graphs of the same call (separate static inputs / private pools) replayed together
    on two streams: the family's scratch must not be shared state."""
    import torch
    xs = case.inputs
    statics = [_clone_input(xs[0]), _clone_input(xs[1 % len(xs)])]
    graphs, outs = [], []
    for st in statics:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=torch.cuda.Stream()):
            outs.append(case.run(st))
        graphs.append(g)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    eager = []
    for x in xs:
        tmp = _clone_input(xs[0])
        case.copy_in(tmp, x)
        eager.append(case.run(tmp).clone())
    torch.cuda.synchronize()
    cur = torch.cuda.current_stream()
    for r in range(rounds):
        pick = [(r + i) % len(xs) for i in range(2)]
        for st, j in zip(statics, pick):
            case.copy_in(st, xs[j])
        for g, s in zip(graphs, streams):
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                g.replay()
        for s in streams:
            cur.wait_stream(s)
        torch.cuda.synchronize()
        for o, j in zip(outs, pick):
            if not bits_equal(o, eager[j]):
                raise Fp4mmaError(f"{MARKER} concurrent replay {r}: output of input {j} differs "
                                  "from eager — shared state between two graphs")


def gate_k5(plist, recs, log=_log) -> dict:
    """K5 over what a GPU gate just compiled, with the lab's strictness (spills fail: K2): every dense dot_scaled
    body and every MoE fc1 / fc2 that ran must carry kind::mxf4nvf4 for sm_120a, the hardware-cvt control its
    cvt. (The MoE run_plan also judges its own kernels at every GPU call, serving's strictness.) Returns
    the gate record."""
    from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
    ran = {r["label"] for r in recs if r["status"] in ("PASS", "FAIL")}
    bodies = {p.spec["body"] for p in plist if p.family == "gemm" and p.label in ran}
    names = [fm.TRITON_NAMES[b] for b in sorted(bodies) if b != fm.CVT]
    if any(p.family == "moe" and p.label in ran for p in plist):
        from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fe
        names += [fe.TRITON_NAMES[k] for k in fe.NATIVE_KERNELS]
    try:
        if names:
            assert_native(names, allow_local=False, log=log)
        if fm.CVT in bodies:
            assert_contains([fm.TRITON_NAMES[fm.CVT]], ["cvt.rn.f16x2.e2m1x2"], "hardware FP4 decode", log=log)
    except Fp4mmaError as exc:
        return {"label": "K5", "family": "all", "status": "FAIL", "detail": str(exc)[:900]}
    return {"label": "K5", "family": "all", "status": "PASS",
            "detail": f"{len(names)} native kernels + {int(fm.CVT in bodies)} cvt control judged on the compiled PTX"}


def run_gate(tier="quick", rt=None, dev=None, replays=6, streams=2, capture=True, guards=True,
             only="", plist=None, builders=None, emit=lambda s: print(s, flush=True)) -> dict:
    """The gate as a function (the lab driver and the CPU tests call it). Returns
    {"verdict": PASS | FAIL | NOT RUN, "runtime", "probes": [...]}. `builders` maps a probe's
    family to its build function (default: the family modules')."""
    rt = rt or CudaRuntime
    dev = dev or ("cuda" if rt is CudaRuntime else "cpu")
    plist = plist if plist is not None else all_probes(tier)
    plist = [p for p in plist if only in p.label]
    if builders is None:
        from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
        builders = {"gemm": fm.build_probe}
        try:
            from suffix_hybrid.helion_kernels import dsv41_moe_fp4mma as fmoe
            builders["moe"] = fmoe.build_probe
        except ImportError:
            pass
    emit(f"{MARKER} fp4mma_capture_replay: {len(plist)} probes, {replays} replays each, runtime "
         f"{rt.name}, capture={capture} guards={guards} streams={streams}")
    recs = []
    for pr in plist:
        emit(f"{MARKER} PROBE {pr.label}: {pr.family} {json.dumps(pr.spec, sort_keys=True)} [{pr.source}]")
        rec = run_probe(pr, builders[pr.family], rt, dev, replays, streams, capture, guards)
        recs.append(rec)
        emit(f"{MARKER} PROBE {pr.label}: {rec['status']} — {rec.get('detail', '')}")
        if rec["status"] == "FAIL" and ("CUDA" in rec["detail"] or "illegal" in rec["detail"]):
            emit(f"{MARKER} sticky CUDA error: stopping at the first faulting probe")
            break
    if rt is CudaRuntime and any(r["status"] in ("PASS", "FAIL") for r in recs):
        k5 = gate_k5(plist, recs, log=lambda s: emit(f"{MARKER} {s}"))
        recs.append(k5)
        emit(f"{MARKER} PROBE K5: {k5['status']} — {k5['detail']}")
    fail = [r for r in recs if r["status"] == "FAIL"]
    ran = [r for r in recs if r["status"] == "PASS"]
    verdict = "FAIL" if fail else ("PASS" if ran else "NOT RUN")
    emit(f"{MARKER} FP4MMA-CAPTURE-REPLAY {verdict} probes {len(ran)} pass / {len(fail)} fail / "
         f"{len(recs) - len(ran) - len(fail)} not run")
    return {"verdict": verdict, "runtime": rt.name, "probes": recs}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fp4mma_probe", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    cr = sub.add_parser("capture_replay", help="the capture/replay boot gate")
    tier = cr.add_mutually_exclusive_group()
    tier.add_argument("--quick", action="store_true", help="a few probes (default)")
    tier.add_argument("--full", action="store_true", help="every shape x bucket x body x split")
    cr.add_argument("--replays", type=int, default=6)
    cr.add_argument("--streams", type=int, default=2, help="2 = also replay two graphs concurrently")
    cr.add_argument("--only", default="", help="run probes whose label contains this")
    cr.add_argument("--no-capture", action="store_true", help="eager only (sanitizer isolation)")
    cr.add_argument("--no-guards", action="store_true")
    cr.add_argument("--json", action="store_true", help="print the result dict as the last line")
    ptx = sub.add_parser("ptx", help="analyze a PTX file (P0 helper)")
    ptx.add_argument("path")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    if a.cmd == "ptx":
        with open(a.path) as f:
            info = analyze_ptx(f.read())
        print(json.dumps({**info, "problems": ptx_problems(info, allow_local=False)}, indent=1))
        return 0
    if a.cmd != "capture_replay":
        parser().print_help()
        return 2
    import torch
    if not torch.cuda.is_available():
        print(f"{MARKER} FP4MMA-CAPTURE-REPLAY NOT RUN: no CUDA device", flush=True)
        return 2
    try:
        require_native_stack()
    except Fp4mmaError as exc:
        print(f"{exc}\n{MARKER} FP4MMA-CAPTURE-REPLAY NOT RUN", flush=True)
        return 2
    install_recorder()
    cc = torch.cuda.get_device_capability()
    from suffix_hybrid import helion_kernels as hk
    print(f"{MARKER} gpu {torch.cuda.get_device_name(0)} cc {cc[0]}.{cc[1]} platform {hk.platform()}",
          flush=True)
    res = run_gate("full" if a.full else "quick", replays=a.replays, streams=a.streams,
                   capture=not a.no_capture, guards=not a.no_guards, only=a.only)
    if a.json:
        print(json.dumps(res), flush=True)
    return {"PASS": 0, "FAIL": 1, "NOT RUN": 2}[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
