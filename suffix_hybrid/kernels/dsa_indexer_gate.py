# SPDX-License-Identifier: Apache-2.0
"""Boot gate ``dsa_indexer_capture_replay``: the DSA indexer lane's decode paths
captured into a CUDA graph on a side stream and replayed with CHANGING inputs
and context lengths, bit-exact against eager launches of the same call.

    python -m suffix_hybrid.kernels.dsa_indexer capture_replay [--quick] [--default-configs]
    compute-sanitizer --tool memcheck --error-exitcode 99 \
        python -m suffix_hybrid.kernels.dsa_indexer capture_replay --quick

V4.1 decode runs the indexer inside the full decode graph (one capture per
batch size, replayed at every context), and silicon history says graph capture
is where lane code breaks (an unresolved illegal memory access in a drafter's
prefill capture, JIT compiles and module loads mid-capture, barriers that
deadlock). What the gate proves, per entry and shape:

 1. warm-up is enough: ``warm_all`` (+ the remap warm-up) is the ONLY thing that
    runs before the capture -- the captured call has never executed eagerly, so
    a kernel the warm-up missed would JIT inside the capture and fail here (the
    ``dsa_capture.launch`` guard turns that into a precise error instead of a
    fault);
 2. nothing syncs or allocates outside torch's allocator inside the capture
    (CUDA refuses either);
 3. replays after rewriting EVERY input in place (q, weights, K pages, block
    tables, per-row lengths incl. 0 / 1 / page and tile edges / beyond the
    logits width) equal an eager launch on the same inputs, bit for bit, on
    every column a consumer reads (top-k sets for the ranking steps);
 4. the f64 spec still holds on the first replay (the oracle of the lab).

V4.1 geometry only (the gate pins SUFFIX_DSA_INDEXER_GEOMETRY=v41): the oxide kernels
are retired for V4.1, so the gate runs Helion bodies keyed fmt=v41 and never imports
the native extension. Entries: ``row`` (Helion per-row decode body), ``group`` (Helion row-group
body), ``cand`` (fused candidate logits + the Triton slot remap, + vLLM's
persistent top-k when vLLM is importable), ``candsel`` (valid-block selection
for the candidate source layer; needs vLLM). Shipped configs are used as
shipped (the pool's real set); ``--default-configs`` forces the bounded Helion
default config on every kernel (the lab, before anything is shipped).
An entry whose kernel has no shipped config reports ``skipped (not shipped ...)``
(the shim wrapper serves the stock path there, exactly as in serving -- the vendored
DeepGEMM verbatim where it runs, as in A1, else the shim's Triton stand-in -- and K5 logs
it per layer) -- never a silent pass: the line says which route ran.
"""
from __future__ import annotations

import os
import sys

MARKER = "[suffix dsa-indexer]"
HEADS = 32
EDGE = (0, 1, 2, 5, 6, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257)
# (requests, rows per request, max context, page, page pad bytes): V4.1 ratio-1 (64 / 128 token
# pages, packed page strides), ratio-2 pages (32 tokens at the block-64 hook), non-multiple-of-G rows
CASES = ((1, 6, 70000, 64, 0), (4, 6, 40000, 128, 128), (3, 5, 30000, 32, 128), (8, 6, 12000, 64, 0),
         (1, 6, 524288, 64, 0))   # the d6 target context
QUICK = ((1, 6, 9000, 64, 0), (3, 5, 6000, 128, 128))
NCAND, CBS, TOPK = 2048, 8, 512


def _lens_for(trial, reqs, ctx_max, n_cols, capacity, gen):
    """Per-request context lengths of one replay: edges first, then a mix of
    random, the logits-width boundary and beyond it."""
    import torch
    base = torch.randint(1, ctx_max + 1, (reqs,), generator=gen)
    for i in range(reqs):
        pick = (trial + i) % 4
        if pick == 0:
            base[i] = EDGE[(trial * 3 + i) % len(EDGE)]
        elif pick == 1:
            base[i] = n_cols + (trial % 3) * 5   # at / past the logits width (the kernels clamp)
        elif pick == 2:
            base[i] = ctx_max
    return base.clamp_max(capacity)     # a context never outruns its block table


class Rig:
    """Static buffers of one captured shape (stable pointers across replays) and
    the in-place refill that changes everything a replay may see."""

    def __init__(self, dev, reqs, nn, ctx_max, page, pad, seed):
        import torch
        self.dev, self.reqs, self.nn, self.page, self.pad = dev, reqs, nn, page, pad
        self.rows = reqs * nn
        self.ctx_max = ctx_max
        self.n_cols = ctx_max + 64
        self.capacity = self.n_cols + 16                  # tables cover a little past the logits width
        self.width = -(-self.capacity // page)            # block-table width (pages)
        self.nb = reqs * self.width + 4                   # + spare pages the tables may point at
        self.stride = page * 132 + pad
        self.back = torch.zeros(self.nb * self.stride, dtype=torch.uint8, device=dev)
        self.cache = self.back.as_strided((self.nb, page, 132), (self.stride, 132, 1))
        self.cache4 = self.cache.unsqueeze(2)
        self.q = torch.zeros((self.rows, HEADS, 128), dtype=torch.float8_e4m3fn, device=dev)
        self.w = torch.zeros((self.rows, HEADS), dtype=torch.float32, device=dev)
        self.cl = torch.zeros(self.rows, dtype=torch.int32, device=dev)
        self.bt = torch.zeros((self.rows, self.width), dtype=torch.int32, device=dev)
        self.cand = torch.full((self.rows, NCAND), -1, dtype=torch.int32, device=dev)
        self.logits = torch.zeros((self.rows, self.n_cols), dtype=torch.float32, device=dev)
        self.gen = torch.Generator().manual_seed(seed)
        self.gdev = torch.Generator(device=dev).manual_seed(seed)
        self.lens = None

    def refill(self, trial):
        """Rewrite every input in place; the pointers never change."""
        import torch
        dev, g, gd = self.dev, self.gen, self.gdev
        nb, page = self.nb, self.page
        kf = (torch.randn((nb * page, 128), generator=gd, device=dev) * 0.5).to(torch.float8_e4m3fn)
        ksc = 2.0 ** torch.randint(-8, -3, (nb * page,), generator=gd, device=dev).float()
        pages = self.back.view(nb, self.stride)
        pages[:, :page * 128] = kf.view(torch.uint8).reshape(nb, page * 128)
        pages[:, page * 128:page * 132] = ksc.view(torch.uint8).reshape(nb, page * 4)
        if self.pad:   # the bytes between pages are garbage a kernel must never read
            pages[:, page * 132:] = torch.randint(0, 256, (nb, self.pad), generator=gd, device=dev,
                                                  dtype=torch.uint8)
        self.q.copy_((torch.randn((self.rows, HEADS, 128), generator=gd, device=dev) * 2).to(torch.float8_e4m3fn))
        self.w.copy_(torch.randn((self.rows, HEADS), generator=gd, device=dev) * 0.02)
        lens = _lens_for(trial, self.reqs, self.ctx_max, self.n_cols, self.capacity, g)
        self.lens = lens
        nn = self.nn
        cl = (lens.unsqueeze(1) - nn + 1 + torch.arange(nn)).clamp_min(0).to(torch.int32).reshape(-1)
        self.cl.copy_(cl.to(self.dev))
        perm = torch.randperm(nb - 4, generator=g)       # disjoint pages per request, like the allocator
        bt = torch.zeros((self.reqs, self.width), dtype=torch.int32)
        o = 0
        for i, n in enumerate(lens.tolist()):
            nbk = min(-(-n // page), self.width)
            bt[i, :nbk] = perm[o:o + nbk].to(torch.int32)
            o += nbk
        self.bt.copy_(bt.repeat_interleave(nn, 0).to(self.dev))
        cand = torch.full((self.rows, NCAND), -1, dtype=torch.int32)
        for r in range(self.rows):
            n = int(cl[r])
            nbl = -(-n // CBS)
            if nbl:
                ids = torch.cat([torch.tensor([nbl - 1]), torch.randperm(nbl - 1, generator=g)[:min(NCAND, nbl) - 1]])
                cand[r, :ids.numel()] = ids[torch.randperm(ids.numel(), generator=g)].to(torch.int32)
        self.cand.copy_(cand.to(self.dev))
        self.logits.copy_(torch.randn((self.rows, self.n_cols), generator=gd, device=dev))

    def live(self):
        """bool [rows, n_cols]: the columns a consumer reads (the kernels leave the rest unwritten)."""
        import torch
        col = torch.arange(self.n_cols, device=self.dev)
        return col[None, :] < torch.clamp(self.cl, 0, self.n_cols).long()[:, None]


# ---------------------------------------------------------------------------
# entries: (run(rig) -> {name: tensor}, how to compare each tensor, route label)
# ---------------------------------------------------------------------------
def _row_route(rig):
    """"helion per-row", or why not (V4.1 geometry: no config means the stock path serves)."""
    from suffix_hybrid import helion_kernels as hk
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_indexer as di
    di.refuse_oxide_for_v41()
    if hk.config(hdi.DECODE, hdi.serving_case(hdi.DECODE, HEADS, rig.rows, fmt=hdi.FMT_V41)) is not None:
        return "helion per-row"
    return None


def entry_row(rig):
    from suffix_hybrid.kernels import dsa_indexer as di
    route = _row_route(rig)
    if route is None:
        return ("not shipped for this rows bucket (the shim wrapper serves the stock path, the baseline's own "
                "kernel: nothing of ours to capture)")

    def run():
        return {"logits": di.paged_logits(rig.q.view(rig.rows, 1, HEADS, 128), rig.cache4, rig.w,
                                          rig.cl.view(rig.rows, 1), rig.bt, rig.n_cols, group=1)}
    return run, {"logits": "window"}, route, None


def entry_group(rig):
    from suffix_hybrid.kernels import dsa_indexer as di

    def run():
        return {"logits": di.paged_logits(rig.q.view(rig.rows, 1, HEADS, 128), rig.cache4, rig.w,
                                          rig.cl.view(rig.rows, 1), rig.bt, rig.n_cols, group=6)}
    if di._use_group(HEADS, rig.rows, 1, 6, None, True):
        route = "helion row-group"
    else:
        route = _row_route(rig)
        if route is None:
            return "no row-group and no per-row config shipped for this rows bucket (stock path serves)"
        route = "row-group NOT shipped for this rows bucket: " + route
    return run, {"logits": "window"}, route, None


def entry_cand(rig):
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_candidates as dc
    topk = _topk(rig.dev)

    def run():
        out = torch.empty((rig.rows, NCAND * CBS), dtype=torch.float32, device=rig.dev)
        hdi.cand_logits(rig.q, rig.w, rig.cache4, rig.bt, rig.cl, rig.cand, out)
        res = {"cand_logits": out}
        if topk is not None:
            lens = hdi.cand_lens(rig.cl, NCAND, CBS).to(torch.int32)
            idx = torch.empty((rig.rows, TOPK), dtype=torch.int32, device=rig.dev)
            topk(out, lens, 1, idx, TOPK, NCAND * CBS)
            dc.cand_remap(idx, rig.cand, rig.cl, CBS)
            res["topk_positions"] = idx
        return res
    return run, {"cand_logits": "cand", "topk_positions": "set"}, "helion fused candidates" \
        + (" + persistent top-k + remap" if topk is not None else " (no vLLM: top-k skipped)"), \
        lambda: dc.warm_remap(rig.dev, CBS, TOPK, NCAND)


def entry_candsel(rig):
    import torch
    from suffix_hybrid.kernels import dsa_candidates as dc
    if -(-rig.n_cols // CBS) < NCAND:
        return None                       # fewer blocks than k: vLLM's own path runs, nothing to gate
    try:
        from vllm.model_executor.kernels.attention.dsa import candidate_blocks as cb
    except Exception:  # noqa: BLE001
        return None
    topk = _topk(rig.dev)
    if topk is None:
        return None
    fast = dc.select_blocks_fn(cb, lambda backend: topk, cb.select_candidate_blocks)

    def run():
        out = torch.full((rig.rows, NCAND), -1, dtype=torch.int32, device=rig.dev)
        fast(rig.logits, None, rig.cl, NCAND, CBS, out)
        return {"blocks": out}
    def warm():
        # vLLM's _block_scores_kernel[FULL_WIDTH=False] is a Triton JIT of its own: compile it for the
        # row-count classes (1 | other | % 16) at this logits width before any capture
        for rows in (1, 2, 16):
            fast(torch.randn((rows, rig.n_cols), device=rig.dev), None,
                 torch.full((rows,), rig.n_cols, dtype=torch.int32, device=rig.dev), NCAND, CBS,
                 torch.full((rows, NCAND), -1, dtype=torch.int32, device=rig.dev))
    return run, {"blocks": "set"}, "valid-block selection + persistent top-k", warm


def _topk(dev):
    """vLLM's persistent_topk without the engine, or None when vLLM is absent."""
    try:
        import torch
        import vllm._custom_ops  # noqa: F401
        ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)

        def topk(logits, lens, nn, out, k, bound):
            torch.ops._C.persistent_topk(logits, lens.reshape(-1, 1).to(torch.int32), out, ws, k, bound)
        return topk
    except Exception:  # noqa: BLE001
        return None


ENTRIES = (("row", entry_row), ("group", entry_group), ("cand", entry_cand), ("candsel", entry_candsel))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def same_bits(a, b, mask):
    """a, b float32 equal bit for bit on the masked columns."""
    import torch
    return bool(torch.equal(a.view(torch.int32)[mask], b.view(torch.int32)[mask]))


def same_sets(a, b):
    return [sorted(x for x in r if x >= 0) for r in a.tolist()] == [sorted(x for x in r if x >= 0) for r in b.tolist()]


def compare(rig, how, graph_out, eager_out):
    """-> (ok, what)"""
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    for name, kind in how.items():
        if name not in graph_out:
            continue
        a, b = graph_out[name], eager_out[name]
        if kind == "window":
            if not same_bits(a, b, rig.live()):
                return False, f"{name}: replay != eager on the window"
        elif kind == "cand":
            col = torch.arange(a.shape[1], device=a.device)
            mask = col[None, :] < hdi.cand_lens(rig.cl.long(), NCAND, CBS)[:, None]
            if not same_bits(a, b, mask):
                return False, f"{name}: replay != eager on the candidate slots"
        elif kind == "set":
            if not same_sets(a.cpu(), b.cpu()):
                return False, f"{name}: replay and eager select different sets"
    return True, ""


# ---------------------------------------------------------------------------
# runners: the CUDA graph, and a CPU stand-in the unit tests drive
# ---------------------------------------------------------------------------
class CudaRunner:
    """Capture on a side stream; replay in place."""

    def __init__(self, dev):
        import torch
        self.torch, self.dev = torch, dev

    def capture(self, run):
        torch = self.torch
        side = torch.cuda.Stream(self.dev)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(self.dev)
        with torch.cuda.graph(graph, stream=side):
            out = run()
        torch.cuda.synchronize(self.dev)
        return graph, out

    def replay(self, graph):
        graph.replay()
        self.torch.cuda.synchronize(self.dev)

    def sync(self):
        self.torch.cuda.synchronize(self.dev)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def run_gate(runner, dev, cases=CASES, trials=4, entries=None, log=print, f64=True):
    """-> list of result lines; raises RuntimeError at the first failure."""
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    lines = []
    for reqs, nn, ctx, page, pad in cases:
        for name, make in (entries or ENTRIES):
            rig = Rig(dev, reqs, nn, ctx, page, pad, seed=ctx + reqs + page)
            rig.refill(0)
            built = make(rig)
            if built is None or isinstance(built, str):
                lines.append(f"capture_replay {name} reqs={reqs}x{nn} page={page}: skipped ("
                             + (built or f"needs vLLM, or width < {NCAND * CBS}") + ")")
                log(lines[-1])
                continue
            run, how, route, warm = built
            if name == "cand" and not hdi.has_config(hdi.CANDK, HEADS, rig.rows, ncand=NCAND, bsz=CBS):
                lines.append(f"capture_replay cand reqs={reqs}x{nn} page={page}: not shipped (dense path serves)")
                log(lines[-1])
                continue
            # warm-up is the ONLY thing before the capture: the captured call never ran eagerly
            hdi.warm_all(dev, rig.cache4, rig.n_cols, rig.width, HEADS, 6,
                         cand=(NCAND, CBS) if name == "cand" else None, v41=True)
            if warm is not None:
                warm()
            graph, out_graph = runner.capture(run)
            for trial in range(trials):
                rig.refill(trial + 1)
                runner.replay(graph)
                eager = run()
                runner.sync()
                ok, what = compare(rig, how, out_graph, eager)
                if not ok:
                    raise RuntimeError(f"{MARKER} DSA-INDEXER CAPTURE_REPLAY FAIL {name} reqs={reqs}x{nn} "
                                       f"page={page} ctx<={ctx} replay {trial}: {what} "
                                       f"(lens {rig.lens.tolist()})")
                if f64 and trial == 0 and "logits" in out_graph and rig.rows <= 64:
                    ok, msg = _f64(rig, out_graph["logits"])
                    if not ok:
                        raise RuntimeError(f"{MARKER} DSA-INDEXER CAPTURE_REPLAY FAIL {name} vs f64: {msg}")
            lines.append(f"capture_replay {name} reqs={reqs}x{nn} page={page}+{pad} ctx<={ctx}: "
                         f"{trials} replays bit-exact vs eager ({route}) OK")
            log(lines[-1])
            del graph, out_graph, rig
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return lines


def _f64(rig, logits):
    """The f64 spec on the window of every row (the kernels leave the rest unwritten)."""
    import torch
    from suffix_hybrid.helion_kernels import dsa_indexer as hdi
    from suffix_hybrid.kernels import dsa_indexer as di
    kv, ks, page, prow, sp = hdi.cache_views(rig.cache4)
    val, mag = hdi.decode_reference(rig.q, rig.w, kv, ks, rig.bt, rig.cl, logits, page, prow, sp, 2)
    inside = ~torch.isnan(val)
    got = logits[inside].double()
    if not bool(torch.isfinite(got).all()):
        return False, "non-finite logits inside the windows"
    err = float(((got - val[inside]).abs() / (mag[inside] + 1e-30)).max()) if got.numel() else 0.0
    return err <= di.TOL, f"max err/A {err:.2e} (tol {di.TOL:.0e})"


def default_configs():
    """Force the bounded Helion default config on every kernel of the lane (and
    one compiled object per body): the lab's pre-ship run of the gate."""
    from suffix_hybrid import helion_kernels as hk
    hk.config = lambda kernel, case, plat=None: {}
    cache = {}

    def compiled(body, kernel, case, **settings):
        if body not in cache:
            hk.ensure_vendor()
            import helion
            settings.setdefault("static_shapes", False)
            cache[body] = helion.kernel(body, config=helion.Config(), **settings)
        return cache[body]
    hk.compiled = compiled


def main(argv=None):
    """CLI: exit 0 on PASS, 1 on any failure (compute-sanitizer adds its own exit code)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    quick = "--quick" in argv
    try:
        import torch
        os.environ.setdefault("SUFFIX_HELION_DSA_INDEXER", "helion")
        os.environ.setdefault("SUFFIX_DSA_INDEXER_GEOMETRY", "v41")   # this gate is the V4.1 one: no oxide anywhere
        if "--default-configs" in argv:
            default_configs()
        dev = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(dev)[0] != 12:
            raise RuntimeError("dsa_indexer_capture_replay is the SM120 gate (cc 12.x only)")
        lines = run_gate(CudaRunner(dev), dev, QUICK if quick else CASES, trials=2 if quick else 4,
                         log=lambda m: print(f"{MARKER} {m}", file=sys.stderr, flush=True))
        print(f"{MARKER} DSA-INDEXER CAPTURE_REPLAY PASS ({len(lines)} entries)", file=sys.stderr, flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"{MARKER} DSA-INDEXER CAPTURE_REPLAY FAIL: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
