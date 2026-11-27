# SPDX-License-Identifier: Apache-2.0
"""Boot gate ``dsv41_attn_capture_replay`` for the Helion DeepSeek-V4.1 attention.

The failure it exists for: an illegal memory access in the drafter's prefill
CUDA-graph capture (g14). Anything that makes a captured kernel differ from its
eager twin -- a pointer bound at first use, a scalar baked from the capture-time
batch, scratch from outside torch's allocator, a compile or a host sync inside the
capture -- shows up here, at boot, instead of as a crash or silent garbage under load.

For every kernel of a loaded route (partial+merge pair and row body, every T bucket and
context class) at the bucket's upper edge and at its lower edge + 1:

  1. warm eagerly (every config is compiled before any capture);
  2. capture ``attend`` on a SIDE stream into a CUDA graph (the sync-debug mode is on
     inside the captured region: any implicit host sync raises);
  3. replay the graph ``trials`` times with CHANGING inputs: the static buffers (q, SWA /
     compressed index lists, live lengths = ctx lengths including 0 and full, sink and the
     pools' bytes) are overwritten in place between replays, exactly what a serving step
     does; the outputs are pre-filled with NaN so a row the graph forgot to write is caught;
  4. each replay must equal the eager run on the same inputs BIT-EXACTLY (the kernels are
     deterministic: no atomics, fixed reduction order), be finite, and (first trials) pass
     the K3 reader gate against the exact f64 reference;
  5. no allocation survives a replay and the output pointer is stable.

Any failure raises (fail loud: the pod must not start with an attention path that is not
capture-safe). Runnable alone, also under ``compute-sanitizer`` (memcheck / racecheck /
initcheck all work on graphs; use --small to keep the run short):

    python -m suffix_hybrid.kernels.dsv41_attn_capture --small
    compute-sanitizer --tool memcheck --print-limit 10 \\
        python -m suffix_hybrid.kernels.dsv41_attn_capture --small --buckets 6,12,32

Without shipped configs the CLI validates the kernels with one default config per body
(--default-config, the default when nothing is shipped), so the gate runs in the kernel lab
before any autotune result exists. ``SUFFIX_DSV41_ATTN_GATE=0`` skips the boot run (the
gate is on whenever the Helion attention is).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

import torch

from suffix_hybrid.helion_kernels import dsv41_attn as D
from suffix_hybrid.helion_kernels import dsv41_attn_ref as R

TAG = "[suffix dsv41_attn_capture_replay]"
GATE_ENV = "SUFFIX_DSV41_ATTN_GATE"


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


class GateFailure(RuntimeError):
    """The attention path is not capture-safe (or replay differs from eager)."""


# ---------------------------------------------------------------------------
# backends: the real CUDA graph, and an eager stand-in the CPU tests drive
# ---------------------------------------------------------------------------
class CudaBackend:
    name = "cuda-graph"

    def __init__(self, device):
        self.device = torch.device(device)
        self.side = torch.cuda.Stream(self.device)

    def sync(self):
        torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def no_host_sync(self):
        prev = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(prev)

    def capture(self, fn):
        """Capture ``fn`` on the side stream; returns a replay callable."""
        self.sync()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.side):
            fn()
        return graph.replay

    def memory(self) -> int:
        return torch.cuda.memory_allocated(self.device)


class EagerBackend:
    """No GPU: 'capture' records the callable and replay re-runs it. It proves the
    gate's own logic (input mutation, NaN prefill, bit-exact compare, failure
    reporting); the CUDA-graph semantics need CudaBackend on silicon."""
    name = "eager-standin"

    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def sync(self):
        pass

    @contextlib.contextmanager
    def no_host_sync(self):
        yield

    def capture(self, fn):
        fn()
        return fn

    def memory(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def gate_sizes(name: str, bucket: int, small: bool) -> list:
    """T values exercised per bucket: its upper edge and its lower edge + 1
    (``small``: only those <= 48, at least one)."""
    sizes = D.warm_sizes(name, bucket)
    keep = sorted({sizes[0], sizes[-1]})
    return ([t for t in keep if t <= 48] or keep[:1]) if small else keep


def _static_inputs(prob: dict, dev):
    """The long-lived device buffers one captured call reads (stable pointers): the
    inputs and the flat bytes of both pools (``*_flat``) behind their vLLM-shaped views."""
    st = {"q": prob["q"].to(dev).clone(),
          "swa_idx": prob["swa_idx"].to(dev).clone()[:, None, :],
          "swa_len": prob["swa_len"].to(dev).clone(),
          "cmp_idx": prob["cmp_idx"].to(dev).clone(),
          "cmp_len": prob["cmp_len"].to(dev).clone(),
          "sink": prob["sink"].to(dev).clone(), "swa": None, "cmp": None}
    for key in ("swa", "cmp"):
        if prob[key] is not None:
            flat, rec, page, stride = prob[key]
            st[key + "_flat"] = flat.to(dev).clone()
            st[key] = D.vllm_view_of(st[key + "_flat"], rec, page, stride)
    return st


def _load_inputs(st: dict, prob: dict) -> None:
    """Overwrite the static buffers in place with a new problem (a serving step)."""
    st["q"].copy_(prob["q"])
    st["swa_idx"].copy_(prob["swa_idx"][:, None, :])
    st["swa_len"].copy_(prob["swa_len"])
    st["cmp_idx"].copy_(prob["cmp_idx"])
    st["cmp_len"].copy_(prob["cmp_len"])
    st["sink"].copy_(prob["sink"])
    for key in ("swa", "cmp"):
        if prob[key] is not None:
            st[key + "_flat"].copy_(prob[key][0])


def _call(route, st, out, ctx):
    return D.attend(route, st["q"], st["swa"], st["swa_idx"], st["swa_len"], st["cmp"], st["cmp_idx"],
                    st["cmp_len"], st["sink"], out, ctx)


def one_case(route: dict, name: str, bucket: int, ctx: str, t: int, trials: int, be, seed: int,
             reference_trials: int = 1) -> dict:
    """Capture + replay one (kernel, bucket, ctx, T); raises GateFailure."""
    heads, swa_cap, topk = route["heads"], route["swa"], route["topk"]
    srec, crec = route["recs"]
    dev = be.device
    mk = lambda i, ref: D.adversarial_problem(  # noqa: E731
        t, heads, swa_cap, srec, crec, seed=seed + 7919 * i, topk=topk or R.TOPK, with_cmp=topk > 0,
        spage=route["pages"][0], cpage=route["pages"][1], sample=24, reference=ref)
    prob0 = mk(0, reference_trials > 0)
    st = _static_inputs(prob0, dev)
    one = {**route, "mode": "helion", "kernels": {D.PARTIAL: {}, D.ROW: {}}}
    one["kernels"][name][(bucket, ctx)] = route["kernels"][name][(bucket, ctx)]
    out_g = torch.zeros(t, heads, R.HD, dtype=torch.bfloat16, device=dev)
    out_e = torch.zeros_like(out_g)
    where = f"{name} T={t} (bucket {bucket}, ctx {ctx}, rec {route['label']}, swa {swa_cap}, topk {topk})"

    with be.no_host_sync():
        _call(one, st, out_e, ctx)                      # 1. eager warm (compile) -- also a sync-free proof
    be.sync()

    def captured():
        with be.no_host_sync():
            _call(one, st, out_g, ctx)
    try:
        replay = be.capture(captured)                   # 2. capture on the side stream
    except Exception as exc:
        raise GateFailure(f"{where}: CUDA graph capture failed: {type(exc).__name__}: {exc}") from exc
    be.sync()
    mem0, ptr0 = be.memory(), out_g.data_ptr()
    for i in range(trials):                             # 3. replay with changing inputs
        prob = prob0 if i == 0 else mk(i, i < reference_trials)
        if i:
            _load_inputs(st, prob)
        out_g.fill_(float("nan"))
        out_e.fill_(float("nan"))
        with be.no_host_sync():
            _call(one, st, out_e, ctx)                  # eager twin on the same inputs
        be.sync()
        replay()
        be.sync()
        if not bool(torch.isfinite(out_g.float()).all()):
            raise GateFailure(f"{where}: replay {i} left non-finite values (rows not written by the graph?)")
        if not torch.equal(out_g.view(torch.int16), out_e.view(torch.int16)):   # 4. bit-exact
            bad = (out_g.view(torch.int16) != out_e.view(torch.int16)).any(-1).any(-1).nonzero().flatten()
            raise GateFailure(f"{where}: replay {i} differs from eager (rows {bad[:8].tolist()}): the captured "
                              "graph is not equivalent to the eager call (stale binding / baked scalar / "
                              "pointer)")
        if prob["ref"] is not None:
            ok, msg = R.check(out_g.cpu()[prob["rows"]], prob["ref"][prob["rows"]])
            if not ok:
                raise GateFailure(f"{where}: replay {i} fails the K3 gate: {msg}")
    if be.memory() != mem0 or out_g.data_ptr() != ptr0:  # 5. nothing allocated / moved by replays
        raise GateFailure(f"{where}: memory_allocated or the output pointer changed across replays")
    return {"kernel": name, "bucket": bucket, "ctx": ctx, "T": t, "trials": trials, "backend": be.name}


def dsv41_attn_capture_replay(route: dict, device=None, *, trials: int = 3, small: bool = False, seed: int = 0,
                              backend=None, only_buckets=None) -> list:
    """The boot gate: every kernel of ``route`` through capture -> replay (see module
    doc). Returns one result dict per case; raises GateFailure on the first failure."""
    be = backend or CudaBackend(device if device is not None else route["device"])
    t0 = time.time()
    results = []
    for name in (D.PARTIAL, D.ROW):
        for (b, cx) in sorted(route["kernels"][name]):
            if only_buckets and b not in only_buckets:
                continue
            for t in gate_sizes(name, b, small):
                results.append(one_case(route, name, b, cx, t, trials, be, seed))
    _log(f"PASS {len(results)} cases, {trials} replays each, backend {be.name}, rec {route['label']}, "
         f"swa {route['swa']}, topk {route['topk']}, {time.time() - t0:.1f}s")
    return results


# ---------------------------------------------------------------------------
# CLI (kernel lab / compute-sanitizer)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {D.PARTIAL: {"block_sizes": [16], "num_warps": 4, "num_stages": 2, "num_splits": 4},
                  D.ROW: {"block_sizes": [16], "num_warps": 4, "num_stages": 2}}


def default_route(heads, swa_cap, topk, recs, device, buckets=None, cmp_page=32):
    """A route with ONE default config per body for every bucket (no shipped configs needed)."""
    import helion

    lab = D.rec_label(*recs, topk)
    kernels = {D.PARTIAL: {}, D.ROW: {}}
    for name in (D.PARTIAL, D.ROW):
        k = helion.kernel(D._BODY[name], config=helion.Config(**DEFAULT_CONFIG[name]), static_shapes=False,
                          **D.SETTINGS)
        for b in D.buckets(name):
            if buckets is None or b in buckets:
                for cx in (["c512k"] if D.case_ctx(b, topk, "c512k") != "any" else ["any"]):
                    kernels[name][(b, cx)] = k
    return D.new_route("helion", heads, swa_cap, topk, recs, kernels, lab, torch.device(device), (32, cmp_page))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rec", choices=("v4", "v41"), default="v4")
    ap.add_argument("--shape", default="128:512", help="swa:topk (128:512 | 128:0 | 192:0)")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--cmp-page", type=int, default=32, choices=(32, 64),
                    help="compressed-pool block tokens: 32 (ratio-2 layers) | 64 (ratio-1 layers)")
    ap.add_argument("--buckets", default="", help="comma list; default all")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--small", action="store_true", help="T <= 48 only (compute-sanitizer)")
    ap.add_argument("--shipped", action="store_true", help="use configs/helion/ instead of one default config")
    a = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print(f"{TAG} needs a CUDA device (CPU tests drive dsv41_attn_capture_replay with EagerBackend)",
              file=sys.stderr)
        return 2
    swa, topk = (int(x) for x in a.shape.split(":"))
    recs = (R.REC_V4, R.REC_V4) if a.rec == "v4" else (R.REC_FP8, R.REC_FP4 if topk else R.REC_FP8)
    buckets = {int(x) for x in a.buckets.split(",") if x} or None
    dev = torch.device("cuda", torch.cuda.current_device())
    if a.shipped:
        os.environ.setdefault(D.KNOB, "auto")
        route = D.load(a.heads, swa, topk, *recs, dev, a.cmp_page)
        if route is None:
            print(f"{TAG} no shipped configs for this geometry; rerun without --shipped", file=sys.stderr)
            return 2
    else:
        route = default_route(a.heads, swa, topk, recs, dev, buckets, a.cmp_page)
        D.warm_and_verify(route, dev)                    # compile + K3 before the captures
    dsv41_attn_capture_replay(route, dev, trials=a.trials, small=a.small, only_buckets=buckets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
