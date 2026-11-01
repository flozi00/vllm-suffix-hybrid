# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 shared expert fused into the routed Helion MoE lane
(SUFFIX_DSV41_MOE_FUSE_SHARED): knobs, eligibility, the serving state, the layer
oracle, the boot capture-replay gate and the K4 bench. Kernels:
suffix_hybrid/helion_kernels/dsv41_moe_shared.py (design, math, long-pole argument).

Knobs (default OFF = the previous plugin, byte for byte):
  SUFFIX_DSV41_MOE_FUSE_SHARED=1          needs SUFFIX_NVFP4_MOE=1 and
      SUFFIX_HELION_NVFP4_MOE=helion (the oxide-free route). Fail closed: every V4 NVFP4 routed layer whose
      shared expert cannot be fused, a missing tuned config for a served M, a failed layer
      oracle or capture-replay gate is a STARTUP ERROR naming the reason.
  SUFFIX_DSV41_MOE_FUSE_SHARED_POLE_TOL=<x>  boot guard against the long pole (default 1.05): the
      first fused layer is timed in CUDA graphs, cold L2, fused vs routed-only + vLLM's own
      overlapped shared MLP, at M=1 and at the ceiling; fused slower than x * the stock
      composition is a startup error naming both numbers (lower the ceiling or switch off).
  SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M=<n>  largest M run fused (default 8 until the lab says;
      1..192, <= SUFFIX_NVFP4_MOE_MAX_M). Above it: routed-only (our lane up to MAX_M, vLLM's
      FlashInfer path above) + vLLM's own shared MLP, overlapped on its aux stream exactly as
      the runner would have (SharedExperts.maybe_forward_async / wait / output), then
      `shared + routed` (the runner's add, bit for bit).

How it hooks in (anchors pinned at vLLM 39983a5e7, refused on drift): MoERunner.__init__ is
wrapped once to hand each SuffixRoutedExperts a weak reference to its runner. At load, after
the oracle, a fused layer takes `runner._shared_experts` OUT of the runner (set to None,
`_forward_entry` re-selected: the runner then neither runs nor adds a shared expert), keeps
the SharedExperts object itself (its aux stream and events exist since the runner's
__init__: nothing is created at the first forward) and its weights are read in place. The
runner's `_maybe_reduce_final_output` all-reduces our `routed_partial + shared_partial`
exactly where it reduced `shared_output + fused_output`: one reduce per layer, the same
[M, H] bf16 partial layout (output_is_reduced False for FlashInfer CUTLASS, checked).

Capture rules (the g14 lesson, dossier glm53-speed-2026-10-11 "g13 / g14": an illegal
address in the drafter's prefill graph capture, suspects: the oxide shared slot and a
Triton dense GEMM with split-K semaphores bound to streams on first use):
  * the fused plan has NO state: no tickets, counters or semaphores; every launch writes
    everything it reads later; its scratch (h, hs, y, the sorted-pair arrays) is torch.empty
    inside the Helion bodies = torch's allocator (the graph pool under capture)
  * no host sync, no allocation of streams / events, no first-use binding in the serving path
    (tests/test_dsv41_moe_shared.py audits the serving functions' AST)
  * every served M's kernels are compiled at load (FusedPlan.warm); a compile inside a capture
    is refused loudly (helion_kernels/dsv41_moe_shared.FusedPlan.__call__)
  * `moe_capture_replay` (boot gate per fused layer and a CLI): capture the layer in a
    torch.cuda.graph on a side stream at several M, replay with CHANGING inputs
    (balanced / skewed / hot / all-off-rank routing, x re-drawn), interleaving the graphs in
    random order, outputs poisoned with NaN before each replay: bit-exact vs eager every time,
    twice per input (determinism). Under compute-sanitizer:
      compute-sanitizer --tool memcheck python -m suffix_hybrid.kernels.dsv41_moe_shared \
          moe_capture_replay        (DSV41_MOE_GATE_MS=1,6,48 shrinks the M set)
    A CPU stand-in graph (replay = run again into the fixed output buffer) runs the same
    logic in tests/test_dsv41_moe_shared.py.

CLI (in-pod, SM120): oracle | bench | moe_capture_replay.
  bench = K4: fused vs (routed-only + the stock MXFP8 shared MLP on an aux stream, inside one
  CUDA graph, vLLM's own kernels), cold L2, balanced + skewed expert loads, M in
  1/6/12/24/48/96/192; prints the table, the shared-expert cost either way and
  `SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M=<n>` (largest M of an unbroken fused win from M=1).
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import os
import sys
import textwrap
import zlib
from unittest import mock

MARKER = "[suffix dsv41-moe-shared]"
KNOB = "SUFFIX_DSV41_MOE_FUSE_SHARED"
MAX_M_ENV = "SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M"
GATE_MS_ENV = "DSV41_MOE_GATE_MS"
POLE_TOL_ENV = "SUFFIX_DSV41_MOE_FUSE_SHARED_POLE_TOL"
DEFAULT_POLE_TOL = 1.05
DEFAULT_MAX_M = 8
CAP_MAX_M = 192
ORACLE_MS = (1, 6, 8, 12, 24, 48, 96, 192)
BENCH_MS = (1, 6, 12, 24, 48, 96, 192)
GATE_MS = (1, 2, 6, 7, 13, 48, 96, 192)  # capture-replay M set (boot gate keeps the first few)
SHARED_REF_TOL = 4e-3     # shared partial vs the f64 spec of this math (bf16 output floor ~1.7e-3)
SHARED_STOCK_TOL = 0.10   # shared partial vs vLLM's module (MXFP8 act quant on both GEMMs)
FULL_STOCK_TOL = 0.15     # fused layer vs FlashInfer routed + the module (gross-error bound)
# the stock kernels whose post-processed layout the bodies read in place
KERNELS_OK = ("FlashInferCutlassMxfp8LinearKernel", "SuffixMxfp8LinearKernel")


def _log(msg: str) -> None:
    print(f"{MARKER} {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# knobs
# ---------------------------------------------------------------------------
def fuse_on() -> bool:
    v = os.environ.get(KNOB, "").strip()
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise RuntimeError(f"{MARKER} {KNOB}={v!r}: expected 0 or 1")


def fuse_max_m(lane_max_m: int) -> int:
    """The fused ceiling: the knob (1..192, an explicit value above the lane's own
    SUFFIX_NVFP4_MOE_MAX_M is an error) or DEFAULT_MAX_M capped by the lane's ceiling."""
    raw = os.environ.get(MAX_M_ENV, "").strip()
    if not raw:
        return min(DEFAULT_MAX_M, lane_max_m)
    if not raw.isdigit() or not 1 <= int(raw) <= CAP_MAX_M:
        raise RuntimeError(f"{MARKER} {MAX_M_ENV}={raw!r}: need an integer in 1..{CAP_MAX_M}")
    if int(raw) > lane_max_m:
        raise RuntimeError(f"{MARKER} {MAX_M_ENV}={raw} exceeds the lane's "
                           f"SUFFIX_NVFP4_MOE_MAX_M={lane_max_m}")
    return int(raw)


def check_env(gate_on: bool, helion_backend: str) -> None:
    """arm()/register() time: the knob needs the lane and a Helion backend."""
    if not fuse_on():
        return
    if not gate_on:
        raise RuntimeError(f"{MARKER} {KNOB}=1 needs SUFFIX_NVFP4_MOE=1")
    if helion_backend != "helion":
        raise RuntimeError(f"{MARKER} {KNOB}=1 needs SUFFIX_HELION_NVFP4_MOE=helion (the fused "
                           f"plan is Helion only and the DeepSeek-V4 route is oxide-free; got "
                           f"{helion_backend!r})")


# ---------------------------------------------------------------------------
# eligibility (pure: plain facts from collect_facts, so CPU tests pin every reason)
# ---------------------------------------------------------------------------
def _r(v: int, a: int) -> int:
    return -(-v // a) * a


def shared_eligibility(s: dict) -> str | None:
    """Why a layer's shared expert cannot be fused, or None."""
    if not s.get("runner"):
        return "no MoERunner reference (the runner hook did not fire for this layer)"
    if not s.get("shared"):
        return "the layer has no shared expert"
    for key, why in (("shared_gate", "gated shared expert (shared_expert_gate)"),
                     ("transform", "routed input/output transform (latent MoE)"),
                     ("sp", "sequence parallel"),
                     ("reduced", "the routed kernel reduces its own output (the runner would "
                                 "all-reduce the shared output separately)"),
                     ("mk_overlap", "shared experts overlapped inside the modular kernel"),
                     ("bias", "shared expert biases"),
                     ("reduce_results", "the shared down_proj all-reduces by itself")):
        if s.get(key):
            return why
    if s.get("dp", 1) != 1:
        return f"data parallel (dp={s.get('dp')})"
    if float(s.get("rsf", 1.0)) != 1.0:
        return (f"runner routed_scaling_factor {s.get('rsf')} != 1.0 (the runner would scale the "
                "fused output, shared included)")
    if not s.get("mlp"):
        return "shared expert is not gate_up_proj -> SiluAndMulWithClamp -> down_proj"
    if s.get("kernel") not in KERNELS_OK:
        return (f"shared linears run on {s.get('kernel')}, not a kernel with FlashInfer CUTLASS's "
                f"MXFP8 layout {KERNELS_OK} (weight e4m3 [N, K], F8_128x4 scales)")
    if s.get("wdtype") != "float8_e4m3fn":
        return f"shared expert weights {s.get('wdtype')} (float8_e4m3fn only)"
    if s.get("sdtype") not in ("uint8", "float8_e8m0fnu"):
        return f"shared expert scales {s.get('sdtype')} (e8m0 bytes only)"
    h, i = s.get("H"), s.get("I")
    if not h or not i or h % 32 or i % 32:
        return f"shared H/I {h}/{i}: whole 32-blocks needed"
    if h != s.get("routed_H"):
        return f"shared hidden {h} != routed hidden {s.get('routed_H')}"
    if s.get("gu_shape") != (2 * i, h) or s.get("dn_shape") != (h, i):
        return (f"shared weight shapes {s.get('gu_shape')} / {s.get('dn_shape')} != "
                f"{(2 * i, h)} / {(h, i)}")
    if s.get("sgu_numel") != _r(2 * i, 128) * _r(h // 32, 4) \
            or s.get("sd_numel") != _r(h, 128) * _r(i // 32, 4):
        return (f"shared scale layout is not the flat F8_128x4 swizzle "
                f"({s.get('sgu_numel')}, {s.get('sd_numel')} bytes)")
    if s.get("limit") is None or float(s["limit"]) != float(s.get("routed_limit") or -1):
        return f"shared swiglu_limit {s.get('limit')} != routed {s.get('routed_limit')}"
    return None


def collect_facts(runner, info: dict):
    """(plain facts for shared_eligibility, the shared MLP or None) from vLLM's objects
    (duck-typed: tests pass fakes)."""
    s = {"runner": runner is not None, "dp": info.get("dp", 1),
         "mk_overlap": bool(info.get("mk_shared_overlap")), "routed_H": info.get("H"),
         "routed_limit": info.get("clamp_limit")}
    sh = getattr(runner, "_shared_experts", None)
    mlp = getattr(sh, "_layer", None)
    s["shared"] = mlp is not None
    if mlp is None:
        return s, None
    mc = runner.moe_config
    s.update(shared_gate=getattr(runner, "shared_expert_gate", None) is not None,
             transform=(getattr(runner, "routed_input_transform", None) is not None
                        or getattr(runner, "routed_output_transform", None) is not None),
             sp=bool(getattr(mc, "is_sequence_parallel", False)),
             reduced=bool(runner._fused_output_is_reduced),
             rsf=float(runner.routed_scaling_factor))
    gu, dn = getattr(mlp, "gate_up_proj", None), getattr(mlp, "down_proj", None)
    wg, wd = getattr(gu, "weight", None), getattr(dn, "weight", None)
    sg, sd = getattr(gu, "weight_scale", None), getattr(dn, "weight_scale", None)
    act = getattr(mlp, "act_fn", None)
    s["mlp"] = (type(act).__name__ == "SiluAndMulWithClamp" and wg is not None and wd is not None
                and wg.dim() == wd.dim() == 2 and sg is not None and sd is not None
                and (getattr(act, "alpha", 1.0), getattr(act, "beta", 0.0)) == (1.0, 0.0))
    if not s["mlp"]:
        return s, None
    kern = [type(getattr(getattr(m, "quant_method", None), "kernel", None)).__name__
            for m in (gu, dn)]
    s.update(kernel=kern[0] if kern[0] == kern[1] else "/".join(kern),
             wdtype=str(wg.dtype).replace("torch.", "") if wg.dtype == wd.dtype else "mixed",
             sdtype=str(sg.dtype).replace("torch.", "") if sg.dtype == sd.dtype else "mixed",
             bias=getattr(gu, "bias", None) is not None or getattr(dn, "bias", None) is not None,
             reduce_results=bool(getattr(dn, "reduce_results", False)),
             limit=float(act.swiglu_limit), H=wg.shape[1], I=wg.shape[0] // 2,
             gu_shape=tuple(wg.shape), dn_shape=tuple(wd.shape),
             sgu_numel=sg.numel(), sd_numel=sd.numel())
    return s, mlp


def validate_weights(wq, sq, what: str) -> None:
    """Fail closed on bytes the bodies would turn into NaN / inf: e4m3 NaN codes in the
    weights, e8m0 scale bytes outside [S_MIN, S_MAX] (255 = NaN; > 246 can overflow bf16)."""
    import torch
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    w8 = wq.view(torch.uint8)
    s8 = sq.view(torch.uint8)
    # the swizzle pads with zero bytes: only the real part is checked by value range, the NaN
    # scan is global (a padding byte is 0)
    if bool(((w8 & 0x7F) == 0x7F).any()):
        raise RuntimeError(f"{MARKER} {what}: e4m3 NaN codes in the shared weights")
    nz = s8[s8 != 0]
    if nz.numel() and (int(nz.max()) > hs.S_MAX):
        raise RuntimeError(f"{MARKER} {what}: e8m0 scale byte {int(nz.max())} > {hs.S_MAX} "
                           "(255 is NaN; larger can overflow bf16)")


# ---------------------------------------------------------------------------
# serving state
# ---------------------------------------------------------------------------
class SharedState:
    """Everything a fused layer needs at forward time; plain attributes only (no lazy
    state: see the capture rules in the module doc)."""

    def __init__(self, *, plan, sh, obj, no_overlap, mlp, hdim, isz, limit, fuse_max, name):
        self.plan, self.sh, self.obj, self.no_overlap = plan, sh, obj, no_overlap
        self.mlp, self.hdim, self.isz, self.limit = mlp, hdim, isz, limit
        self.fuse_max, self.name = fuse_max, name

    def fused(self, x) -> bool:
        import torch
        return (x.dim() == 2 and x.dtype == torch.bfloat16 and x.shape[1] == self.hdim
                and 1 <= x.shape[0] <= self.fuse_max)

    def forward(self, routed, cfg, x, topk_weights, topk_ids):
        """routed(x, tw, ids) = the layer's routed-only path (our lane or vLLM's)."""
        import torch
        if self.fused(x):
            # one dtype per argument = one Triton specialization (the warm-up used int32 / f32)
            tw = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
            ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.int()
            return self.plan(x.contiguous(), ids.contiguous(), tw.contiguous(), cfg["w13"],
                             cfg["w13_sf"], cfg["g1"], cfg["w2"], cfg["w2_sf"], cfg["g2"],
                             cfg["a1g"], cfg["a2g"], cfg["id_base"], self.sh)
        return self.split(routed, x, topk_weights, topk_ids)

    def split(self, routed, x, topk_weights, topk_ids):
        """Routed-only + vLLM's shared MLP in the runner's own order: launch the shared
        expert (aux stream when vLLM would), run the routed experts, join, add."""
        started = self.obj.maybe_forward_async(x)
        if not started:
            self.obj.forward(x, self.no_overlap)
        out = routed(x, topk_weights, topk_ids)
        if started:
            self.obj.wait()
        return self.obj.output + out


# ---------------------------------------------------------------------------
# the serving-path audit (CPU test): no host sync, no stream / event creation
# ---------------------------------------------------------------------------
FORBIDDEN_ATTRS = {"item", "tolist", "cpu", "numpy", "synchronize", "nonzero", "unique",
                   "wait_stream", "record_stream", "sleep", "Stream", "Event", "CUDAGraph",
                   "manual_seed", "pin_memory"}
FORBIDDEN_NAMES = {"sleep"}


def audit_function(fn) -> list:
    """Forbidden constructs in a serving function's source: [(line, what)]."""
    src = textwrap.dedent(inspect.getsource(fn))
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            bad.append((node.lineno, "." + node.attr))
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad.append((node.lineno, node.id))
    return bad


def serving_functions() -> list:
    """The functions that run per forward in a fused layer."""
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    return [SharedState.forward, SharedState.split, SharedState.fused, hs.FusedPlan.__call__,
            hs.FusedPlan.kernels, hs.FusedPlan.key, hs.run_plan_sh, hs.lane_case_sh]


# ---------------------------------------------------------------------------
# capture / replay gate (CUDA graph on a side stream; a CPU stand-in for the logic)
# ---------------------------------------------------------------------------
class _CpuGraph:
    """CPU stand-in for torch.cuda.CUDAGraph in tests: capture() runs the function once into a
    fixed output buffer, replay() runs it again into the SAME buffer (stable 'pointer')."""

    def __init__(self):
        self.fn = None
        self.out = None

    def capture(self, fn):
        self.fn = fn
        self.out = fn().clone()
        return self.out

    def replay(self):
        self.out.copy_(self.fn())


@contextlib.contextmanager
def _quiet_capture():
    """torch.cuda.graph.__enter__ runs gc.collect() and empty_cache(): seconds in a big worker,
    x layers x graphs at boot. vLLM's own CUDAGraphWrapper patches both out for the same reason."""
    import torch
    with mock.patch("gc.collect", lambda *a, **k: 0), \
            mock.patch.object(torch.cuda, "empty_cache", lambda: None):
        yield


def _capture(run, sx, sids, stw, device, cpu_graph: bool, warm: bool):
    """-> (graph, static output). CUDA: an optional warm-up call on the side stream (warm=False
    for code that must already be warm: an unwarmed kernel then fails INSIDE the capture),
    then torch.cuda.graph on that stream."""
    import torch
    if cpu_graph or device.type != "cuda":
        g = _CpuGraph()
        return g, g.capture(lambda: run(sx, sids, stw))
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    if warm:
        with torch.cuda.stream(side):
            run(sx, sids, stw)
        torch.cuda.synchronize(device)
    g = torch.cuda.CUDAGraph()
    with _quiet_capture(), torch.cuda.graph(g, stream=side):
        out = run(sx, sids, stw)
    return g, out


def capture_replay(run, make_batch, ms, *, rounds: int = 3, device=None, cpu_graph: bool = False,
                   warm=True, kinds=("balanced", "skewed", "hot", "dead")) -> str:
    """The moe_capture_replay gate. run(x, ids, tw) -> out [M, H] bf16 (the layer's serving
    call); make_batch(m, seed, kind) -> (x, ids, tw) with kind in `kinds` ('dead' = every
    token routed off-rank). Captures one graph per M on static buffers, then for `rounds`
    rounds replays EVERY graph in a shuffled order with fresh inputs (copied into the static
    buffers) after poisoning the static output with NaN; each replay must equal the eager
    call on the same inputs bit for bit, and a second replay must equal the first. RuntimeError
    names the first mismatch. `warm` (bool or m -> bool): run `run` once on the side stream
    before capturing that M (code vLLM owns, e.g. its shared MLP, is warmed by vLLM's own dummy
    runs in production; OUR kernels must be warm already: warm=False makes a compile inside the
    capture the failure it would be). Returns the PASS line."""
    import random

    import torch
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(0)
    graphs = {}
    for m in ms:
        x0, ids0, tw0 = make_batch(m, m, kinds[0])
        sx, sids, stw = x0.clone(), ids0.clone(), tw0.clone()
        g, out = _capture(run, sx, sids, stw, device, cpu_graph,
                          warm(m) if callable(warm) else warm)
        graphs[m] = (g, out, sx, sids, stw)
    n = 0
    for r in range(rounds):
        order = list(graphs)
        rng.shuffle(order)
        for m in order:
            g, out, sx, sids, stw = graphs[m]
            kind = kinds[(r + m) % len(kinds)]
            x, ids, tw = make_batch(m, 1000 * (r + 1) + m, kind)
            eager = run(x.clone(), ids.clone(), tw.clone()).clone()
            sx.copy_(x)
            sids.copy_(ids)
            stw.copy_(tw)
            first = None
            for rep in range(2):
                out.fill_(float("nan"))
                g.replay()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                if not torch.equal(out, eager):
                    bad = int((out != eager).sum())
                    raise RuntimeError(
                        f"{MARKER} moe_capture_replay FAIL: M={m} round {r} kind {kind} replay "
                        f"{rep}: {bad} of {out.numel()} elements differ from the eager call "
                        f"(nan in graph output: {bool(torch.isnan(out.float()).any())})")
                if first is None:
                    first = out.clone()
                elif not torch.equal(first, out):
                    raise RuntimeError(f"{MARKER} moe_capture_replay FAIL: M={m} round {r}: "
                                       "two replays on the same inputs differ (non-deterministic)")
            n += 1
    return (f"{MARKER} moe_capture_replay PASS: {len(ms)} graphs (M={','.join(map(str, ms))}) x "
            f"{rounds} rounds x 2 replays, {n} input sets, bit-exact vs eager, "
            f"{'CPU stand-in' if cpu_graph or device.type != 'cuda' else 'torch.cuda.graph'}")


def gate_ms(max_m: int) -> tuple:
    """The M set of the gate: DSV41_MOE_GATE_MS (comma list) or GATE_MS, <= max_m."""
    raw = os.environ.get(GATE_MS_ENV, "").strip()
    ms = tuple(int(v) for v in raw.split(",") if v.strip()) if raw else GATE_MS
    return tuple(m for m in ms if 1 <= m <= max_m) or (1,)


# ---------------------------------------------------------------------------
# batches, the layer oracle, preparation of one layer
# ---------------------------------------------------------------------------
def batch_factory(cfg: dict, hdim: int, dev):
    """make_batch(m, seed, kind) for capture_replay / oracle / warm-up on a lane cfg (tensors
    and scalars of nvfp4_moe's per-layer dict): activations at the layer's calibrated range,
    GLOBAL routing ids; kind: balanced | skewed | hot | dead (every token off-rank)."""
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    base, eg, k = cfg["id_base"], cfg["E_global"], cfg["topk"]
    local = cfg["w13"].shape[0]
    amax = 2688.0 / cfg["a1g"]

    def make(m, seed, kind="balanced"):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = (torch.randn(m, hdim, generator=gen) * (amax / 4.5)).to(dev, torch.bfloat16)
        dist = kind if kind in nm.ROUTING_DISTS else "balanced"
        dead = True if kind == "dead" else (None if local < eg else False)
        ids, tw = nm.rand_routing(m, eg, k, dev, seed, base, local, dead, dist)
        return x, ids, tw
    return make


def real_shared_spec(sh, hdim: int, isz: int, limit: float) -> dict:
    """The f64 spec operands of this layer's REAL shared weights (dequantized, unswizzled)."""
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    wgu, sgu, wd, sd = sh
    return dict(deq_gu=hs._deq(wgu, sgu, 2 * isz, hdim), deq_d=hs._deq(wd, sd, hdim, isz),
                isz=isz, hdim=hdim, limit=float(limit))


def layer_oracle(name, cfg, info, state, sp_real, routed, parent, make_batch, dev, *,
                 seed=0, ms=None) -> str:
    """The fused layer on its REAL weights, fatal on a mismatch:
      * at every M <= the fused ceiling: vs the f64 spec (routed `moe_ref` + the shared spec of
        sp_real) within REF_TOL, finite, tokens with no local expert = exactly the shared S
      * the shared partial alone (every token routed off-rank): vs the f64 spec within
        SHARED_REF_TOL and vs vLLM's own module within SHARED_STOCK_TOL
      * vs the stock composition (FlashInfer routed `parent` + the module) within FULL_STOCK_TOL
      * M above the ceiling (the split path): routed-only + the module's output, exactly the
        runner's `shared + routed`
    `routed(x, tw, ids)` / `parent(x, tw, ids)` / `state.mlp(x)` are the layer's callables."""
    import torch
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    from suffix_hybrid.kernels import nvfp4_moe as nm
    base, local = cfg["id_base"], info["E"]
    ms = [m for m in (ms or ORACLE_MS) if m <= state.fuse_max] or [1]
    worst = {"ref": 0.0, "sh": 0.0, "stock": 0.0, "sh_stock": 0.0}
    last = ""
    for m in ms:
        x, ids, tw = make_batch(m, seed + m, "balanced")
        lid = nm.to_local(ids, base, local)
        out = state.forward(routed, cfg, x, tw, ids).float()
        ref = nm.moe_ref(cfg, x, lid, tw) + hs.shared_ref(sp_real, x)
        r_ref = nm._rel(out, ref)
        stock = (parent(x.clone(), tw.clone(), ids.clone()) + state.mlp(x.clone())).float()
        r_stock = nm._rel(out, stock)
        xd, idsd, twd = make_batch(m, seed + 500 + m, "dead")
        sh_only = state.forward(routed, cfg, xd, twd, idsd).float()
        sh_ref = hs.shared_ref(sp_real, xd)
        r_sh = nm._rel(sh_only, sh_ref)
        r_sh_stock = nm._rel(sh_only, state.mlp(xd.clone()).float())
        for k, v in (("ref", r_ref), ("sh", r_sh), ("stock", r_stock), ("sh_stock", r_sh_stock)):
            worst[k] = max(worst[k], v)
        last = (f"M={m} rel_vs_spec={r_ref:.2e} shared_rel_vs_spec={r_sh:.2e} "
                f"shared_rel_vs_module={r_sh_stock:.2e} rel_vs_stock={r_stock:.2e}")
        if not (r_ref <= nm.REF_TOL and r_sh <= SHARED_REF_TOL and r_sh_stock <= SHARED_STOCK_TOL
                and r_stock <= FULL_STOCK_TOL and bool(torch.isfinite(out).all())
                and bool(torch.isfinite(sh_only).all())):
            raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL {name} {last} (bounds: spec "
                               f"{nm.REF_TOL}, shared spec {SHARED_REF_TOL}, shared module "
                               f"{SHARED_STOCK_TOL}, stock {FULL_STOCK_TOL}) — refusing to serve "
                               f"with {KNOB}=1")
    m_split = min(state.fuse_max + 1, cfg["max_m"] + 1)
    x, ids, tw = make_batch(m_split, seed + 900, "balanced")
    got = state.split(routed, x, tw, ids)
    want = routed(x.clone(), tw.clone(), ids.clone()) + state.mlp(x.clone())
    if not torch.equal(got, want):
        raise RuntimeError(f"{MARKER} LAYER ORACLE FAIL {name}: the split path (M={m_split}) is not "
                           "the runner's shared + routed bit for bit")
    return (f"{MARKER} LAYER ORACLE PASS {name} fused M<={state.fuse_max} I_sh={state.isz} "
            f"worst rel_vs_spec={worst['ref']:.2e} shared_vs_spec={worst['sh']:.2e} "
            f"shared_vs_module={worst['sh_stock']:.2e} vs_stock={worst['stock']:.2e}; split path "
            f"exact (last {last})")


def pole_tol() -> float:
    raw = os.environ.get(POLE_TOL_ENV, "").strip()
    try:
        v = float(raw) if raw else DEFAULT_POLE_TOL
    except ValueError:
        v = 0.0
    if not v >= 1.0:
        raise RuntimeError(f"{MARKER} {POLE_TOL_ENV}={raw!r}: need a number >= 1.0")
    return v


def long_pole_verdict(rows: dict, tol: float) -> str | None:
    """rows {M: (fused us, routed-only + stock shared us)} -> the error text of the first M where
    the fused plan is slower than tol x the stock composition, or None."""
    for m in sorted(rows):
        f, b = rows[m]
        if f > tol * b:
            return (f"fused shared expert is the long pole at M={m}: {f:.1f} us vs routed-only + "
                    f"vLLM's overlapped shared MLP {b:.1f} us (x{f / b:.2f} > {tol:g}); lower "
                    f"{MAX_M_ENV} (the K4 `bench` prints the ceiling), or set {KNOB}=0")
    return None


def long_pole_gate(state, cfg, routed, make, dev, *, timer=None, iters: int = 30) -> str:
    """Boot guard (first fused layer): fused vs the split composition at M=1 and at the
    ceiling, in CUDA graphs with a cold L2 before every replay (nvfp4_moe._graph_us), on this
    layer's real weights and this node. Raises when fused loses by more than pole_tol()."""
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    timer = timer or nm._graph_us
    flush = None
    if dev.type == "cuda":
        l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)) \
            or (128 << 20)
        buf = torch.zeros(2 * l2 // 4, dtype=torch.float32, device=dev)
        flush = lambda: buf.sum()
    rows = {}
    for m in sorted({1, state.fuse_max}):
        x, ids, tw = make(m, 4242 + m, "balanced")
        rows[m] = (timer(lambda s: state.forward(routed, cfg, x, tw, ids), dev, iters, flush),
                   timer(lambda s: state.split(routed, x, tw, ids), dev, iters, flush))
    bad = long_pole_verdict(rows, pole_tol())
    line = ", ".join(f"M={m}: fused {f:.1f} us vs stock composition {b:.1f} us"
                     for m, (f, b) in sorted(rows.items()))
    if bad:
        raise RuntimeError(f"{MARKER} LONG POLE GATE FAIL ({line}): {bad}")
    return f"{MARKER} long-pole gate PASS ({line}; tolerance x{pole_tol():g})"


def install_runner_hook(layer_cls) -> None:
    """MoERunner.__init__ post-hook: our routed layer keeps a weak reference to its runner
    (the shared MLP and the final add live there). Refuses on drift of the pinned anchors.
    Idempotent (a second call finds the wrapper and returns)."""
    import functools
    import weakref
    try:
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    except ImportError as exc:
        raise RuntimeError(f"{MARKER} {KNOB}=1: this vLLM has no MoERunner ({exc})") from exc
    orig = MoERunner.__init__
    if getattr(orig, "_sfx_hooked", False):
        return
    params = set(inspect.signature(orig).parameters)
    need = {"routed_experts", "shared_experts", "routed_scaling_factor", "shared_expert_gate"}
    if not need <= params or not callable(getattr(MoERunner, "_select_forward", None)):
        raise RuntimeError(f"{MARKER} {KNOB}=1: MoERunner drifted from vLLM 39983a5e7 "
                           f"(__init__ lacks {sorted(need - params)} or _select_forward is gone)")

    @functools.wraps(orig)
    def init(self, *a, **kw):
        orig(self, *a, **kw)
        layer = getattr(self, "routed_experts", None)
        if isinstance(layer, layer_cls):
            layer._sfx_runner = weakref.ref(self)

    init._sfx_hooked = True
    MoERunner.__init__ = init


def detach_from_runner(runner) -> None:
    """The runner stops running / adding a shared expert (ours does, inside forward_modular):
    `_shared_experts` -> None (the _modules key stays: safe mid named_modules()) and the forward
    entry re-selected (moe_forward instead of moe_forward_shared)."""
    runner._shared_experts = None
    runner._forward_entry = runner._select_forward()


def prepare_layer(layer, cfg: dict, info: dict, routed, parent) -> SharedState:
    """Load-time work of SUFFIX_DSV41_MOE_FUSE_SHARED for one SuffixRoutedExperts `layer`
    (cfg = its lane dict after the routed oracle): eligibility, weight checks, the fused plan
    (a missing tuned config for a served M raises), warm-up of every served M, the layer oracle,
    the capture-replay gate; only then the runner lets go of the shared expert. routed(x, tw,
    ids) = the routed-only path, parent(x, tw, ids) = vLLM's FlashInfer path. Raises, never
    falls back."""
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    from suffix_hybrid.kernels import nvfp4_moe as nm
    name = str(layer.layer_name)
    ref = getattr(layer, "_sfx_runner", None)
    runner = ref() if ref is not None else None
    facts, mlp = collect_facts(runner, info)
    why = shared_eligibility(facts)
    if why is not None:
        raise RuntimeError(f"{MARKER} layer {name}: {KNOB}=1 but its shared expert cannot be "
                           f"fused: {why}")
    gu, dn = mlp.gate_up_proj, mlp.down_proj
    sh = hs.sh_views(gu.weight.data, gu.weight_scale.data, dn.weight.data, dn.weight_scale.data)
    validate_weights(sh[0], sh[1], f"{name} gate_up_proj")
    validate_weights(sh[2], sh[3], f"{name} down_proj")
    hdim, isz = facts["H"], facts["I"]
    fmax = fuse_max_m(cfg["max_m"])
    plan = hs.FusedPlan(info["E"], hdim, info["I"], cfg["topk"], cfg["limit"], cfg["E_global"],
                        isz, fmax, lambda m: nm.tune_for(m, cfg.get("tune"), cfg["fused"])[0])
    dev = layer.w13_weight.device
    make = batch_factory(cfg, hdim, dev)

    def run_m(m):  # compile this M's kernels now; never inside a capture
        x, ids, tw = make(m, 77 + m, "balanced")
        plan(x, ids, tw, cfg["w13"], cfg["w13_sf"], cfg["g1"], cfg["w2"], cfg["w2_sf"], cfg["g2"],
             cfg["a1g"], cfg["a2g"], cfg["id_base"], sh)

    pairs = plan.warm(run_m)
    obj = runner._shared_experts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
    state = SharedState(plan=plan, sh=sh, obj=obj, no_overlap=SharedExpertsOrder.NO_OVERLAP,
                        mlp=mlp, hdim=hdim, isz=isz, limit=facts["limit"], fuse_max=fmax,
                        name=name)
    sp_real = real_shared_spec(sh, hdim, isz, facts["limit"])
    seed = zlib.crc32(name.encode()) % 10007
    # the oracle calls the shared module (module forward hooks must not count as the first forward)
    nm._state["in_oracle"] = True
    try:
        msg = layer_oracle(name, cfg, info, state, sp_real, routed, parent, make, dev, seed=seed)
        _log(msg + f" ({pairs} (bucket, plan) pairs warm)")
        # the first fused layer runs the whole gate (every M of GATE_MS, 3 rounds); the others
        # re-prove their own weights / pointers at the fused ceiling, the first split M and M=1
        first = nm._state.get("shared_fused", 0) == 0
        mm = gate_ms(cfg["max_m"])[:6] if first else tuple(sorted({1, fmax, fmax + 1} & set(
            range(1, cfg["max_m"] + 1))))
        _log(capture_replay(lambda x, ids, tw: state.forward(routed, cfg, x, tw, ids),
                            make, mm, rounds=3 if first else 1, device=dev,
                            warm=lambda m: m > fmax))
        if first:
            _log(long_pole_gate(state, cfg, routed, make, dev))
    finally:
        nm._state["in_oracle"] = False
    del sp_real
    detach_from_runner(runner)
    nm._state["shared_fused"] = nm._state.get("shared_fused", 0) + 1
    return state


# ---------------------------------------------------------------------------
# K4: fused vs routed-only + the stock shared MLP on the aux stream (in-pod, SM120)
# ---------------------------------------------------------------------------
def fuse_table(name: str, rows: dict) -> tuple:
    """rows {M: {dist: (fused us, stock-composition us, routed-only us)}} -> (text, ceiling).
    M wins when fused < routed-only + stock shared (aux stream) on EVERY dist (1 % counts);
    the ceiling = the largest M whose whole range from the bottom of the grid wins
    (SUFFIX_DSV41_MOE_FUSE_SHARED_MAX_M: above it the lane runs the stock composition)."""
    lines = [f"FUSE TABLE {name} (cold L2, CUDA graph; fused vs routed-only + stock shared MLP on "
             "the aux stream; shared cost = arm - routed-only)"]
    run, best = True, None
    for m in sorted(rows):
        win = all(f < b for f, b, _ in rows[m].values())
        run = run and win
        best = m if run else best
        lines.append(f"  M={m:<4} " + "  ".join(
            f"{d}: fused {f:.1f} stock {b:.1f} routed {r:.1f} us (x{f / b:.3f}; shared cost "
            f"{f - r:+.1f} vs {b - r:+.1f})" for d, (f, b, r) in sorted(rows[m].items()))
            + f"  {'WIN' if win else 'LOSS'}")
    lines.append(f"  -> {MAX_M_ENV}={best}" if best else
                 "  -> fused loses already at the smallest M: keep vLLM's overlapped shared MLP "
                 f"({KNOB}=0)")
    return "\n".join(lines), best


def stock_shared(sp: dict):
    """vLLM's shared MLP as FlashInferCutlassMxfp8LinearKernel runs it on SM120 (vllm
    model_executor/kernels/linear/mxfp8/flashinfer.py apply_weights, DeepseekV4MLP): dynamic
    MXFP8 quant of x -> mm_mxfp8 -> silu_and_mul_with_clamp -> quant -> mm_mxfp8, on the SAME
    synthetic weights and swizzled scales the fused plan reads. In-pod only (vLLM, FlashInfer)."""
    import torch
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    try:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.utils import flashinfer as vllm_flashinfer
    except ImportError as exc:
        raise RuntimeError(f"{MARKER} the stock shared-MLP baseline needs vLLM + FlashInfer "
                           f"({exc}); run the bench in a lab pod") from exc
    wgu, sgu, wd, sd = hs.sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])
    limit = float(sp["limit"])

    def lin(x, w, s):
        xq, xs = mxfp8_e4m3_quantize(x.view(-1, x.shape[-1]), is_sf_swizzled_layout=True)
        return vllm_flashinfer.mm_mxfp8(xq, w.t(), xs, s, out_dtype=x.dtype, backend="cutlass")

    def run(x):
        gu = lin(x, wgu, sgu)
        h = torch.empty(x.shape[0], gu.shape[1] // 2, dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul_with_clamp(h, gu, limit, 1.0, 0.0)
        return lin(h, wd, sd)
    return run


def _synthetic(dev):
    """(case, routed problem with its Helion op, shared problem, sh views) at the V4.1 EP8 shapes."""
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    from suffix_hybrid.kernels import nvfp4_moe as nm
    case = nm.DS_EP
    p = nm._dev_problem(dev, case, seed=1)
    isz = hs.sh_inter_of({"tp": 8}, {"moe_intermediate": case[2], "experts_global": case[0]})
    sp = hs.make_shared_problem(case[1], isz, seed=1, device=dev, limit=nm.DS_EP[8])
    return case, p, sp, hs.sh_views(sp["wgu"], sp["sgu"], sp["wd"], sp["sd"])


def _plan(case, ms, untuned_ok=True):
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    from suffix_hybrid.kernels import nvfp4_moe as nm
    e_global, hdim, idim, topk = case[:4]
    local = e_global // case[5]
    isz = hs.sh_inter_of({"tp": 8}, {"moe_intermediate": idim, "experts_global": e_global})
    return hs.FusedPlan(local, hdim, idim, topk, case[8], e_global, isz, max(ms),
                        lambda m: nm.tune_for(m)[0], untuned_ok=untuned_ok)


def _arms(dev, ms):
    """The three timed callables of one layer (fused, stock composition, routed-only) plus the
    pieces tests and the gates reuse. Events and the aux stream are made here, once, outside any
    capture (like vLLM's SharedExperts)."""
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    case, p, sp, sh = _synthetic(dev)  # V4.1 is oxide-free: no `_native`, no cubin, no workspace
    e_global, hdim, idim, topk = case[:4]
    base, local = p["id_base"], e_global // case[5]
    plan = _plan(case, ms)
    stock = stock_shared(sp)
    aux, ev_in, ev_out = torch.cuda.Stream(dev), torch.cuda.Event(), torch.cuda.Event()

    def routed(x, ids, tw, s):
        return nm.run_ours(None, p, x, ids, tw, nm.NO_WS, s.cuda_stream)

    def fused(x, ids, tw, s):
        return plan(x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"], p["g2"],
                    p["a1g"], p["a2g"], base, sh)

    def composed(x, ids, tw, s):
        ev_in.record(s)
        with torch.cuda.stream(aux):
            ev_in.wait(aux)
            sh_out = stock(x)
            ev_out.record(aux)
        r = routed(x, ids, tw, s)
        ev_out.wait(s)
        return sh_out + r

    return dict(case=case, p=p, sp=sp, sh=sh, plan=plan, fused=fused, composed=composed,
                routed=routed, base=base, local=local)


def bench(ms=BENCH_MS, iters: int = 30, dists=("balanced", "skewed")) -> dict:
    """K4 (module doc). Returns {M: {dist: (fused, composed, routed) us}}."""
    import torch
    from suffix_hybrid.kernels import nvfp4_moe as nm
    dev = torch.device("cuda", torch.cuda.current_device())
    a = _arms(dev, ms)
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 128 << 20)) \
        or (128 << 20)
    buf = torch.zeros(2 * l2 // 4, dtype=torch.float32, device=dev)
    flush = lambda: buf.sum()
    case = a["case"]
    e_global, hdim, topk = case[0], case[1], case[3]
    rows = {}
    for m in ms:
        rows[m] = {}
        for dist in dists:
            ids, tw = nm.rand_routing(m, e_global, topk, dev, seed=100 + m, dead=False,
                                      base=a["base"], local=a["local"], dist=dist)
            x = torch.randn(m, hdim, device=dev).bfloat16()
            t = {k: nm._graph_us(lambda s, k=k: a[k](x, ids, tw, s), dev, iters, flush)
                 for k in ("fused", "composed", "routed")}
            rows[m][dist] = (t["fused"], t["composed"], t["routed"])
            _log(f"bench M={m} {dist}: fused {t['fused']:.1f} us, routed-only + stock shared "
                 f"(aux stream) {t['composed']:.1f} us, routed-only {t['routed']:.1f} us "
                 f"-> x{t['fused'] / t['composed']:.3f}")
    text, best = fuse_table("DeepSeek-V4.1 EP8 rank", rows)
    _log(text)
    return rows


def oracle() -> str:
    """Synthetic EP8 rank: the fused plan (Helion default configs where none are tuned) vs the
    f64 spec at every ORACLE_MS, balanced / skewed / hot / all-off-rank; the shared partial
    alone vs the spec. Fatal on mismatch."""
    import torch
    from suffix_hybrid.helion_kernels import dsv41_moe_shared as hs
    from suffix_hybrid.kernels import nvfp4_moe as nm
    dev = torch.device("cuda", torch.cuda.current_device())
    case, p, sp, sh = _synthetic(dev)
    e_global, hdim, idim, topk = case[:4]
    base, local = p["id_base"], e_global // case[5]
    plan = _plan(case, ORACLE_MS)
    cfg = dict(p, E_global=e_global, topk=topk, id_base=base)
    make = batch_factory(cfg, hdim, dev)
    lines = []
    for m in ORACLE_MS:
        for kind in ("balanced", "skewed", "hot", "dead"):
            x, ids, tw = make(m, m, kind)
            lid = nm.to_local(ids, base, local)
            out = plan(x, ids, tw, p["w13"], p["w13_sf"], p["g1"], p["w2"], p["w2_sf"], p["g2"],
                       p["a1g"], p["a2g"], base, sh).float()
            ref = nm.moe_ref(p, x, lid, tw) + hs.shared_ref(sp, x)
            r = nm._rel(out, ref)
            lines.append(f"M={m} {kind}: rel_vs_spec={r:.2e}")
            if not (r <= nm.REF_TOL and bool(torch.isfinite(out).all())):
                raise RuntimeError(f"{MARKER} ORACLE FAIL {lines[-1]} (bound {nm.REF_TOL})")
            if kind == "dead":
                rs = nm._rel(out, hs.shared_ref(sp, x))
                if rs > SHARED_REF_TOL:
                    raise RuntimeError(f"{MARKER} ORACLE FAIL M={m}: shared partial alone "
                                       f"rel_vs_spec={rs:.2e} > {SHARED_REF_TOL}")
    for ln in lines:
        _log(ln)
    return f"{MARKER} ORACLE PASS ({len(lines)} cases)"


def moe_capture_replay() -> str:
    """CLI gate (module doc): the fused plan, and the stock composition (aux-stream fork/join
    inside the capture), on the synthetic EP8 rank."""
    import torch
    dev = torch.device("cuda", torch.cuda.current_device())
    ms = gate_ms(192)
    a = _arms(dev, ms)
    case = a["case"]
    cfg = dict(a["p"], E_global=case[0], topk=case[3], id_base=a["base"])
    make = batch_factory(cfg, case[1], dev)
    cur = lambda: torch.cuda.current_stream(dev)  # the CAPTURING stream inside a capture
    # the plan is warmed exactly like the lane warms it at load; the capture must not compile
    a["plan"].warm(lambda m: a["fused"](*make(m, 5, "balanced"), cur()))
    out = [capture_replay(lambda x, ids, tw: a["fused"](x, ids, tw, cur()), make, ms,
                          device=dev, warm=False)]
    out.append(capture_replay(lambda x, ids, tw: a["composed"](x, ids, tw, cur()), make, ms,
                              device=dev))
    return "\n".join(out)


def main(argv=None) -> int:
    mode = (argv or sys.argv[1:] or ["oracle"])[0]
    try:
        if mode in ("oracle", "all"):
            print(oracle(), file=sys.stderr, flush=True)
        if mode in ("moe_capture_replay", "all"):
            print(moe_capture_replay(), file=sys.stderr, flush=True)
        if mode in ("bench", "all"):
            bench()
        return 0
    except Exception as exc:  # noqa: BLE001 — the gate's verdict is the exit code
        print(f"{MARKER} {mode.upper()} FAIL: {type(exc).__name__}: {exc}", file=sys.stderr,
              flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
