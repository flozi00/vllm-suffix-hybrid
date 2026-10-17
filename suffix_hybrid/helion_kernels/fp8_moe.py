# SPDX-License-Identifier: Apache-2.0
"""Helion port of kernels-oxide/fp8_moe: the FP8 block-scaled (e4m3 weights,
128x128 f32 weight blocks, dynamic 1x128 activation groups) routed-experts
decode MoE — ONE persistent Helion launch for the whole layer, phases split
by hl.barrier() (grid-wide, cooperative launch):

  1  route        topk_ids -> this rank's local ids (id - id_base, outside
                  [0, E) = off-rank) -> pair_list sorted by (expert, pair id)
                  (= moe_route: active experts ascending, pairs ascending),
                  per-expert cnt/off, slot_expert = active experts ascending
                  then -1. Rank-by-counting, no atomics: deterministic.
  2  fc1          grid (slot, I tile): x quant per (token, 128-K group) IN
                  the K loop (moe_quant_rows / the route_quant half), gate +
                  up dots share the quantized A tile, per-128-K-block rescale
                  acc += dot * s_x * s_w (Triton order), h = silu(gate) * up
                  -> f32 h [P, I].
  3  fc2          grid (slot, H tile): h quant per (pair, 128 group) IN the K
                  loop (moe_fc1_quant's quant half), same block rescale,
                  y = topk_w * acc -> f32 y [P, H].
  4  combine      out[t] = bf16(sum_k y[t*topk + k]) over local k, ascending
                  k (moe_combine); a token with no local expert is exact 0.

Ported = every oxide entry (moe_route[_quant], moe_quant_rows, moe_fc1[_quant],
moe_fc2[_combine], moe_combine) as one launch. Difference by design: oxide's
fused fc1 owns a whole 128-col h group per CTA to quantize h in-CTA; here
each fc2 program re-quantizes the h rows it reads (deterministic, bit-equal
between programs), which frees the fc1/fc2 N tiles for the autotuner.

Numerics = suffix_hybrid/kernels/fp8_moe.py's contract (f64 spec moe_ref,
REF_TOL): same quant formulas (x: eps 1e-10, no floor; h: the caller's
eps/floor, quant_params), the pow2 rounding exact (log2/exp2 seed corrected
by exact x2 / x0.5 steps), f32 between the quantizations, fixed-order f32
combine. Triton's fp32 `/` is div.full (<= 2 ulp, like vLLM's own Triton
quant), oxide uses div.rn: a scale may differ by 1 ulp, invisible at REF_TOL.

Tunables: fc1 / fc2 N tiles (hl.register_block_size(16, 128): a tile stays
inside one 128-row weight-scale block), the jagged expert-row tiles (dot
rows >= 16), route tiles, K fixed at the 128 scale block (num_stages
pipelines it), num_warps/stages, persistent SM multiplier, loop orders,
indexing / eviction per load.

Serving: SUFFIX_HELION_FP8_MOE (helion_kernels.backend) inside the enabled
SUFFIX_FP8_MOE lane — fp8_moe.helion_op builds a ServingOp, a drop-in for
_native.fp8_moe_cuda (same call signature) that runs the tuned kernel of
bucket(M) and the oxide op for buckets without a shipped config (auto) —
helion with any missing bucket fails at load. Lab: KERNELS below (one entry,
baseline = the oxide op on the same problem, the whole layer vs the whole
layer).
"""
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.kernels import fp8_moe as fm
from suffix_hybrid.kernels import nvfp4_moe as nm

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU / test hosts: the module stays importable, bodies unused
    helion = hl = None

KERNEL = "fp8_moe"
MARKER = "[suffix helion fp8-moe]"
BUCKETS = (1, 2, 4, 8, 16, 32, 64)  # decode tokens per call (c1..c64)


# ---------------------------------------------------------------------------
# the Helion body (module level, undecorated: the lab and compiled() wrap it)
# ---------------------------------------------------------------------------
def fp8_moe(x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
            w13: torch.Tensor, w13_s: torch.Tensor, w2: torch.Tensor, w2_s: torch.Tensor,
            out: torch.Tensor, id_base: int, slots: int, h_eps: float, h_smin: float,
            pow2: float) -> torch.Tensor:
    """Routed-experts forward of one rank into `out` bf16 [M, H].

    x bf16 [M, H]; topk_ids int [M, K] GLOBAL ids; topk_w f32 [M, K];
    w13 e4m3 [E, 2I, H] ([gate; up]); w13_s f32 [E, 2I/128, H/128];
    w2 e4m3 [E, H, I]; w2_s f32 [E, H/128, I/128]; id_base = this rank's
    first expert; slots = min(E, M*K); (h_eps, h_smin) = fp8_moe.quant_params
    of the h stage; pow2 = 1.0 for UE8M0 activation scales, else 0.0
    (arithmetic blend s + pow2*(s2 - s), exact by Sterbenz: s <= s2 <= 2s)."""
    m = x.size(0)
    hdim = hl.specialize(x.size(1))
    topk = hl.specialize(topk_ids.size(1))
    e_cnt = hl.specialize(w13.size(0))
    idim = hl.specialize(w2.size(2))
    pairs = m * topk
    ids = topk_ids.reshape(pairs)
    tw = topk_w.reshape(pairs)
    # weights as flat rows: an expert's tile is a row gather e*rows + tile.index
    # (a 0-d data-dependent index next to tile dims does not lower)
    w13r = w13.view(e_cnt * 2 * idim, hdim)
    w2r = w2.view(e_cnt * hdim, idim)
    s13v = w13_s.view(e_cnt, 2, idim // 128, hdim // 128)
    pair_list = torch.empty([pairs], dtype=torch.int32, device=x.device)
    slot_expert = torch.empty([e_cnt], dtype=torch.int32, device=x.device)
    cnt = torch.empty([e_cnt], dtype=torch.int32, device=x.device)
    off = torch.empty([e_cnt], dtype=torch.int32, device=x.device)
    h_ws = torch.empty([pairs, idim], dtype=torch.float32, device=x.device)
    y_ws = torch.empty([pairs, hdim], dtype=torch.float32, device=x.device)
    bn1 = hl.register_block_size(16, 128)
    bn2 = hl.register_block_size(16, 128)

    # 1a route: sorted position of every pair = #pairs before it in (key, p)
    # order; key = local id, off-rank pairs = E (sorted last, never served)
    for tile_p in hl.tile(pairs):
        kp = ids[tile_p].to(torch.int32) - id_base
        kp = torch.where((kp >= 0) & (kp < e_cnt), kp, e_cnt)
        pos = hl.zeros([tile_p], dtype=torch.int32)
        for tile_q in hl.tile(pairs):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_cnt), kq, e_cnt)
            before = (kq[None, :] < kp[:, None]) | (
                (kq[None, :] == kp[:, None]) & (tile_q.index[None, :] < tile_p.index[:, None]))
            before = before & (tile_q.index < pairs)[None, :]
            pos = pos + torch.sum(before.to(torch.int32), dim=1).to(torch.int32)
        pair_list[pos] = tile_p.index
    # 1b route: per expert cnt / off (= #pairs with a smaller key), the slot
    # list (one program, E <= 256 by the lane's eligibility)
    for tile_e in hl.tile(e_cnt, block_size=256):
        c = hl.zeros([tile_e], dtype=torch.int32)
        lt = hl.zeros([tile_e], dtype=torch.int32)
        for tile_q in hl.tile(pairs):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_cnt), kq, e_cnt)
            vq = (tile_q.index < pairs)[None, :]
            c = c + torch.sum(((kq[None, :] == tile_e.index[:, None]) & vq).to(torch.int32),
                              dim=1).to(torch.int32)
            lt = lt + torch.sum(((kq[None, :] < tile_e.index[:, None]) & vq).to(torch.int32),
                                dim=1).to(torch.int32)
        ev = tile_e.index
        c = torch.where(ev < e_cnt, c, 0)
        act = (c > 0).to(torch.int32)
        rank = torch.cumsum(act, 0) - act
        n_act = torch.sum(act)
        # active experts -> slots [0, n_act) ascending, the rest -> -1 tail
        slot_expert[torch.where(c > 0, rank, n_act + ev - rank)] = torch.where(c > 0, ev, -1)
        cnt[tile_e] = c
        off[tile_e] = lt
    hl.barrier()

    # 2 fc1 + silu*up, x quantized per (token, 128-K group) in the K loop
    for tile_s, tile_i in hl.tile([slots, idim], block_size=[1, bn1]):
        e = slot_expert[tile_s.begin]
        if e >= 0:
            start = off[e]
            n = cnt[e]
            g_rows = e * (2 * idim) + tile_i.index
            for tile_r in hl.tile(n):
                pr = pair_list[start + tile_r.index]
                tok = pr // topk
                gate = hl.zeros([tile_r, tile_i], dtype=torch.float32)
                up = hl.zeros([tile_r, tile_i], dtype=torch.float32)
                for tile_k in hl.tile(hdim, block_size=128):
                    xv = x[tok, tile_k].to(torch.float32)
                    sx = torch.clamp_min(torch.amax(torch.abs(xv), dim=1), 1e-10) / 448.0
                    p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(sx, 1e-10))))
                    p2 = torch.where(p2 < sx, p2 * 2.0, p2)
                    p2 = torch.where(p2 * 0.5 >= sx, p2 * 0.5, p2)
                    sx = sx + pow2 * (p2 - sx)
                    xq = torch.clamp(xv / sx[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
                    pg = hl.dot(xq, w13r[g_rows, tile_k].T, out_dtype=torch.float32)
                    pu = hl.dot(xq, w13r[g_rows + idim, tile_k].T, out_dtype=torch.float32)
                    gate = gate + pg * sx[:, None] * s13v[e, 0, tile_i.begin // 128, tile_k.id]
                    up = up + pu * sx[:, None] * s13v[e, 1, tile_i.begin // 128, tile_k.id]
                h_ws[pr, tile_i] = gate * torch.sigmoid(gate) * up
    hl.barrier()

    # 3 fc2 * topk_w, h quantized per (pair, 128 group) in the K loop
    for tile_s, tile_h in hl.tile([slots, hdim], block_size=[1, bn2]):
        e = slot_expert[tile_s.begin]
        if e >= 0:
            start = off[e]
            n = cnt[e]
            h_rows = e * hdim + tile_h.index
            for tile_r in hl.tile(n):
                pr = pair_list[start + tile_r.index]
                acc = hl.zeros([tile_r, tile_h], dtype=torch.float32)
                for tile_k in hl.tile(idim, block_size=128):
                    hv = h_ws[pr, tile_k]
                    sh = torch.clamp_min(
                        torch.clamp_min(torch.amax(torch.abs(hv), dim=1), h_eps) / 448.0, h_smin)
                    p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(sh, 1e-10))))
                    p2 = torch.where(p2 < sh, p2 * 2.0, p2)
                    p2 = torch.where(p2 * 0.5 >= sh, p2 * 0.5, p2)
                    sh = sh + pow2 * (p2 - sh)
                    hq = torch.clamp(hv / sh[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
                    pd = hl.dot(hq, w2r[h_rows, tile_k].T, out_dtype=torch.float32)
                    acc = acc + pd * sh[:, None] * w2_s[e, tile_h.begin // 128, tile_k.id]
                y_ws[pr, tile_h] = acc * tw[pr].to(torch.float32)[:, None]
    hl.barrier()

    # 4 combine: local k ascending, f32 -> bf16 once
    for tile_t, tile_o in hl.tile([m, hdim]):
        tot = hl.zeros([tile_t, tile_o], dtype=torch.float32)
        for k in hl.static_range(topk):
            p = tile_t.index * topk + k
            lid = ids[p].to(torch.int32) - id_base
            ok = (lid >= 0) & (lid < e_cnt)
            tot = torch.where(ok[:, None], tot + y_ws[p, tile_o], tot)
        out[tile_t, tile_o] = tot.to(torch.bfloat16)
    return out


def launch_args(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, out, ue8m0: bool, id_base: int):
    """The body's argument tuple for one call (the oxide op's inputs)."""
    h_eps, h_smin = fm.quant_params(bool(ue8m0), "h")
    slots = min(w13.shape[0], x.shape[0] * topk_ids.shape[1])
    return (x, topk_ids, topk_w, w13, w13_s, w2, w2_s, out, int(id_base), slots, float(h_eps),
            float(h_smin), 1.0 if ue8m0 else 0.0)


# ---------------------------------------------------------------------------
# serving: drop-in for _native.fp8_moe_cuda, per-M bucket backend choice
# ---------------------------------------------------------------------------
def layer_case(cfg: dict, numtokens: int) -> dict:
    """The lab case (configs/helion/fp8_moe/<platform>.json key) for a layer."""
    return {"experts": cfg["E"], "hidden": cfg["H"], "intermediate": cfg["I"],
            "topk": cfg["topk"], "numtokens": numtokens}


def served_buckets(max_m: int) -> list:
    return sorted({hk.bucket(m, BUCKETS) for m in range(1, max_m + 1)})


def warm_ms(max_m: int) -> list:
    """Ms the load-time oracle runs so every served bucket compiles before
    CUDA-graph capture (M=1 too: Helion's dynamic shapes specialize size 1)."""
    return sorted({1} | {min(b, max_m) for b in served_buckets(max_m)})


class ServingOp:
    """Same call signature as _native.fp8_moe_cuda. Per call: the Helion
    kernel tuned for bucket(M), else (auto, no shipped config = oxide won or
    untuned) the oxide op with the identical arguments. Workspace, mode and
    stream_ptr are oxide-only: Helion allocates its scratch (graph-pool under
    capture) and launches on torch's current stream (= the stream the lane
    passes)."""

    def __init__(self, cfg: dict, mode: str):
        self.kernels = {}
        missing = []
        for b in served_buckets(cfg["max_m"]):
            case = layer_case(cfg, b)
            if hk.config(KERNEL, case) is None:
                missing.append(b)
            else:
                self.kernels[b] = hk.compiled(fp8_moe, KERNEL, case)
        if mode == "helion" and missing:
            raise RuntimeError(
                f"{MARKER} {fm.HELION_ENV}=helion but no tuned {KERNEL} config on "
                f"{hk.platform()} for numtokens {missing} ({hk.case_key(layer_case(cfg, missing[0]))}"
                f"...): tune it in the autotune lab, or use {fm.HELION_ENV}=auto")
        self.native = None
        if missing:
            from suffix_hybrid import oxide_kernels
            self.native = oxide_kernels.native()
        self.summary = (f"helion numtokens {sorted(self.kernels)}, oxide {missing or '-'} "
                        f"({fm.HELION_ENV}={mode}, {hk.platform()})")

    def kernel(self, m: int):
        return self.kernels.get(hk.bucket(m, BUCKETS))

    def __call__(self, x, topk_ids, topk_w, w13, w13_s, w2, w2_s, ws_aq, ws_as, ws_inter, ws_hq,
                 ws_hs, ws_y, ws_route, out, ue8m0, id_base, mode, stream_ptr):
        kern = self.kernel(x.shape[0])
        if kern is None:
            return self.native.fp8_moe_cuda(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, ws_aq, ws_as,
                                            ws_inter, ws_hq, ws_hs, ws_y, ws_route, out, ue8m0,
                                            id_base, mode, stream_ptr)
        kern(*launch_args(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, out, ue8m0, id_base))
        return None


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def cases(model: dict) -> list:
    """Per-rank shapes x decode buckets; [] unless an FP8 128x128-block MoE
    the lane serves (E <= 256, H and per-rank I multiples of 128). model_dims
    knows TP, not EP: experts = every expert (TP-sharded layout)."""
    e, h, i, k = (model.get(n) for n in ("experts", "hidden", "moe_intermediate", "topk"))
    if (model.get("quant_format") != "fp8_block" or list(model.get("block") or []) != [128, 128]
            or not (e and h and i and k) or not 1 <= e <= 256 or h % 128 or i % 128):
        return []
    dims = {"E": e, "H": h, "I": i, "topk": k}
    return [layer_case(dims, m) for m in BUCKETS]


def make(case: dict, seed: int) -> tuple:
    """Realistic layer: vLLM Fp8MoEMethod layout from fp8_moe.make_problem
    (per-block amax scales), uniform distinct top-k routing, softmax weights."""
    dev = torch.device("cuda", torch.cuda.current_device())
    m, e, k = case["numtokens"], case["experts"], case["topk"]
    p = fm.make_problem(e, case["hidden"], case["intermediate"], device=dev, seed=seed)
    ids, tw = nm.rand_routing(m, e, k, dev, seed=1000 + seed)
    gen = torch.Generator(device="cpu").manual_seed(2000 + seed)
    x = torch.randn(m, case["hidden"], generator=gen).to(dev, torch.bfloat16)
    out = torch.empty(m, case["hidden"], dtype=torch.bfloat16, device=dev)
    return launch_args(x, ids, tw, p["w13"], p["w13_s"], p["w2"], p["w2_s"], out, False, 0)


def reference(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, out, id_base, slots, h_eps, h_smin,
              pow2):
    """The lane's f64 spec (fp8_moe.moe_ref) on the same problem."""
    p = dict(w13=w13, w13_s=w13_s, w2=w2, w2_s=w2_s, ue8m0=pow2 > 0.5)
    return fm.moe_ref(p, x, nm.to_local(topk_ids, id_base, w13.shape[0]), topk_w)


def check(out, ref):
    rel = nm._rel(out.float(), ref)
    ok = rel <= fm.REF_TOL and bool(torch.isfinite(out.float()).all())
    return ok, f"rel_vs_f64_spec {rel:.2e} (gate {fm.REF_TOL:.0e}, the lane oracle's REF_TOL)"


def make_baseline(case: dict, seed: int) -> tuple:
    from suffix_hybrid import oxide_kernels
    oxide_kernels.ensure_loaded(fm.FAMILY)
    x, ids, tw, w13, w13_s, w2, w2_s, out = make(case, seed)[:8]
    ws = fm.workspace(x.device, x.shape[0], ids.shape[1], case["hidden"], case["intermediate"],
                      case["experts"])
    return x, ids, tw, w13, w13_s, w2, w2_s, ws, out, nm.launch_mode(x.shape[0])


def baseline(x, ids, tw, w13, w13_s, w2, w2_s, ws, out, mode):
    """The oxide op exactly as the lane launches it (plan per launch_mode)."""
    from suffix_hybrid import oxide_kernels
    oxide_kernels.native().fp8_moe_cuda(x, ids, tw, w13, w13_s, w2, w2_s, *ws, out, False, 0,
                                        mode, torch.cuda.current_stream().cuda_stream)
    return out


KERNELS = [
    {"name": KERNEL, "body": fp8_moe, "cases": cases, "make": make, "reference": reference,
     "check": check, "make_baseline": make_baseline, "baseline": baseline,
     "baseline_name": "oxide fp8_moe",
     # every PTX .entry of kernels-oxide/fp8_moe (all three launch plans)
     "replaces": [r"^moe_(route|route_quant|quant_rows|fc1|fc1_quant|fc2|fc2_combine|combine)$"]},
]
