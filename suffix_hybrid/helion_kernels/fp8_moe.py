# SPDX-License-Identifier: Apache-2.0
"""Helion port of kernels-oxide/fp8_moe: the FP8 block-scaled (e4m3 weights,
128x128 f32 weight blocks, dynamic 1x128 activation groups) routed-experts
decode MoE (suffix_hybrid/kernels/fp8_moe.py) — one Helion kernel per oxide
launch, chained on the stream like the oxide plans (no grid barrier: a
persistent spin-barrier deadlocks once its grid exceeds co-residency, lab
2026-10-17):

  fp8_moe_route_quant  (= moe_route_quant) three independent roots:
      route   pair_list sorted by (local expert, pair id) by rank counting
              (= moe_route's order; off-rank pairs last), per-expert cnt /
              off, slot_expert = active experts ascending then -1
      quant x vLLM per_token_group_quant_fp8 per (token, 128 cols): eps
              1e-10, s = amax/448 (pow2: UE8M0 rounding) -> xq e4m3, xs f32
  fp8_moe_fc1          (= moe_fc1_quant's GEMM) grid (slot, I tile): the
      expert's rows, gate + up dots share the A tile, per-128-K-block rescale
      acc += dot * s_x * s_w (Triton order), h = silu(gate) * up -> f32 [P, I]
  fp8_moe_fc2          (= moe_fc2) grid (slot, H tile): h quantized per (pair,
      128 group) in the K loop (moe_fc1_quant's quant half: the lane's h-stage
      eps / floor), same rescale, y = topk_w * acc -> f32 [P, H]
  fp8_moe_combine      (= moe_combine) out[t] = bf16(sum_k y[t*topk + k]), local
      k ascending; a token with no local expert is exact 0
  fp8_moe_fc2_combine  (= moe_fc2_combine) token-major GEMV per (token, H
      tile): h quant + fc2 + combine, k ascending (decode M <= FUSED_MAX_M)

Plans (nvfp4_moe.launch_mode per M, both backends): mode 2 (fused3) =
route_quant, fc1, fc2_combine; else (front4) route_quant, fc1, fc2, combine.
h quant moved from fc1 to its consumers so the fc1 N tile is free (oxide's
fc1_quant must own a whole 128-col group per CTA).

Numerics = the lane's contract (f64 spec moe_ref, REF_TOL): the oxide quant
formulas, pow2 rounding exact (log2/exp2 seed corrected by exact x2 / x0.5),
f32 between quantizations, fixed-order f32 combine; deterministic. Triton's
fp32 `/` is div.full (<= 2 ulp, like vLLM's Triton quant; oxide div.rn): a
scale may differ by 1 ulp, invisible at REF_TOL.

Tunables (explicit bounds keep configs compiling fast): route tiles 16..128,
x-quant rows 1..16, fc1/fc2 N tiles 16..128 (a tile stays inside one 128-row
weight-scale block), expert-row tiles 16..64 (dot rows >= 16), combine
tiles, K fixed at the 128 scale block (num_stages pipelines it), warps,
stages, pid type, indexing / eviction per load.

Serving: SUFFIX_HELION_FP8_MOE (helion_kernels.backend) inside the enabled
SUFFIX_FP8_MOE lane — fp8_moe.helion_op builds a ServingOp, a drop-in for
_native.fp8_moe_cuda (same signature): per M the plan's kernels when every
one is tuned for bucket(M), else (auto) the oxide op; helion fails at load.

Lab: one KERNELS entry per launch, baseline = the matching oxide launch
alone: src/fp8_moe_oxide.rs reads bits 8-13 of `mode` as a skip mask
(launch i of the plan skipped when bit SKIP_SHIFT + i is set, plan = mode & 0xFF),
like nvfp4_moe's tune bits 24-29.
"""
import functools

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

MARKER = "[suffix helion fp8-moe]"
BUCKETS = (1, 2, 4, 8, 16, 32, 64)  # decode tokens per call (c1..c64)
SKIP_SHIFT = 8  # oxide baseline: launch i skipped when mode bit 8+i is set
ROUTE_QUANT, FC1, FC2, COMBINE, FC2_COMBINE = (
    "fp8_moe_route_quant", "fp8_moe_fc1", "fp8_moe_fc2", "fp8_moe_combine",
    "fp8_moe_fc2_combine")
PLANS = {"front4": (ROUTE_QUANT, FC1, FC2, COMBINE), "fused3": (ROUTE_QUANT, FC1, FC2_COMBINE)}


# ---------------------------------------------------------------------------
# Helion bodies (module level, undecorated: the lab and compiled() wrap them;
# literals only inside, they are traced)
# ---------------------------------------------------------------------------
def fp8_moe_route_quant(x: torch.Tensor, topk_ids: torch.Tensor, e_count: int, id_base: int,
                        pow2: float):
    """x bf16 [M, H], topk_ids int [M, K] GLOBAL -> (xq e4m3 [M, H], xs f32
    [M, H/128], pair_list i32 [P], slot_expert i32 [E], cnt i32 [E], off i32
    [E]); E = e_count <= 256 local experts from id_base; pow2 1.0 = UE8M0."""
    m = x.size(0)
    hdim = hl.specialize(x.size(1))
    topk = hl.specialize(topk_ids.size(1))
    pairs = m * topk
    ids = topk_ids.reshape(pairs)
    xq = torch.empty([m, hdim], dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty([m, hdim // 128], dtype=torch.float32, device=x.device)
    pair_list = torch.empty([pairs], dtype=torch.int32, device=x.device)
    slot_expert = torch.empty([e_count], dtype=torch.int32, device=x.device)
    cnt = torch.empty([e_count], dtype=torch.int32, device=x.device)
    off = torch.empty([e_count], dtype=torch.int32, device=x.device)
    bp = hl.register_block_size(16, 128)
    bq = hl.register_block_size(16, 128)
    bq2 = hl.register_block_size(16, 64)
    bt = hl.register_block_size(1, 16)

    # sorted position of every pair = #pairs before it in (key, p) order;
    # key = local id, off-rank pairs = E (sorted last, never served)
    for tile_p in hl.tile(pairs, block_size=bp):
        kp = ids[tile_p].to(torch.int32) - id_base
        kp = torch.where((kp >= 0) & (kp < e_count), kp, e_count)
        pos = hl.zeros([tile_p], dtype=torch.int32)
        for tile_q in hl.tile(pairs, block_size=bq):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_count), kq, e_count)
            before = (kq[None, :] < kp[:, None]) | (
                (kq[None, :] == kp[:, None]) & (tile_q.index[None, :] < tile_p.index[:, None]))
            before = before & (tile_q.index < pairs)[None, :]
            pos = pos + torch.sum(before.to(torch.int32), dim=1).to(torch.int32)
        pair_list[pos] = tile_p.index
    # per expert cnt / off (= #pairs with a smaller key) + the slot list: one
    # program over all E (<= 256, the lane's eligibility)
    for tile_e in hl.tile(e_count, block_size=256):
        c = hl.zeros([tile_e], dtype=torch.int32)
        lt = hl.zeros([tile_e], dtype=torch.int32)
        for tile_q in hl.tile(pairs, block_size=bq2):
            kq = ids[tile_q].to(torch.int32) - id_base
            kq = torch.where((kq >= 0) & (kq < e_count), kq, e_count)
            vq = (tile_q.index < pairs)[None, :]
            c = c + torch.sum(((kq[None, :] == tile_e.index[:, None]) & vq).to(torch.int32),
                              dim=1).to(torch.int32)
            lt = lt + torch.sum(((kq[None, :] < tile_e.index[:, None]) & vq).to(torch.int32),
                                dim=1).to(torch.int32)
        ev = tile_e.index
        c = torch.where(ev < e_count, c, 0)
        act = (c > 0).to(torch.int32)
        rank = torch.cumsum(act, 0) - act
        n_act = torch.sum(act)
        # active experts -> slots [0, n_act) ascending, the rest -> -1 tail
        slot_expert[torch.where(c > 0, rank, n_act + ev - rank)] = torch.where(c > 0, ev, -1)
        cnt[tile_e] = c
        off[tile_e] = lt
    # x quant per (token, 128-col group)
    for tile_t, tile_g in hl.tile([m, hdim], block_size=[bt, 128]):
        xv = x[tile_t, tile_g].to(torch.float32)
        s = torch.clamp_min(torch.amax(torch.abs(xv), dim=1), 1e-10) / 448.0
        p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(s, 1e-10))))
        p2 = torch.where(p2 < s, p2 * 2.0, p2)
        p2 = torch.where(p2 * 0.5 >= s, p2 * 0.5, p2)
        s = s + pow2 * (p2 - s)  # exact blend (Sterbenz: s <= p2 <= 2s)
        xq[tile_t, tile_g] = torch.clamp(xv / s[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
        xs[tile_t, tile_g.id] = s
    return xq, xs, pair_list, slot_expert, cnt, off


def fp8_moe_fc1(xq: torch.Tensor, xs: torch.Tensor, pair_list: torch.Tensor,
                slot_expert: torch.Tensor, cnt: torch.Tensor, off: torch.Tensor,
                w13: torch.Tensor, w13_s: torch.Tensor, topk: int, slots: int) -> torch.Tensor:
    """h f32 [P, I] = silu(gate) * up for every local pair (off-rank rows
    unwritten); w13 e4m3 [E, 2I, H] = [gate; up], w13_s f32 [E, 2I/128,
    H/128]; slots = min(E, P)."""
    hdim = hl.specialize(xq.size(1))
    e_cnt = hl.specialize(w13.size(0))
    idim = hl.specialize(w13.size(1)) // 2
    pairs = pair_list.size(0)
    # weights as flat rows: an expert's tile is the row gather e*rows + tile.index
    # (a 0-d data-dependent index next to tile dims does not lower)
    w13r = w13.view(e_cnt * 2 * idim, hdim)
    s13v = w13_s.view(e_cnt, 2, idim // 128, hdim // 128)
    h = torch.empty([pairs, idim], dtype=torch.float32, device=xq.device)
    bn = hl.register_block_size(16, 128)
    br = hl.register_block_size(16, 64)
    for tile_s, tile_i in hl.tile([slots, idim], block_size=[1, bn]):
        e = slot_expert[tile_s.begin]
        if e >= 0:
            start = off[e]
            n = cnt[e]
            g_rows = e * (2 * idim) + tile_i.index
            for tile_r in hl.tile(n, block_size=br):
                pr = pair_list[start + tile_r.index]
                tok = pr // topk
                gate = hl.zeros([tile_r, tile_i], dtype=torch.float32)
                up = hl.zeros([tile_r, tile_i], dtype=torch.float32)
                for tile_k in hl.tile(hdim, block_size=128):
                    a = xq[tok, tile_k]
                    sa = xs[tok, tile_k.id]
                    pg = hl.dot(a, w13r[g_rows, tile_k].T, out_dtype=torch.float32)
                    pu = hl.dot(a, w13r[g_rows + idim, tile_k].T, out_dtype=torch.float32)
                    gate = gate + pg * sa[:, None] * s13v[e, 0, tile_i.begin // 128, tile_k.id]
                    up = up + pu * sa[:, None] * s13v[e, 1, tile_i.begin // 128, tile_k.id]
                h[pr, tile_i] = gate * torch.sigmoid(gate) * up
    return h


def fp8_moe_fc2(h: torch.Tensor, pair_list: torch.Tensor, slot_expert: torch.Tensor,
                cnt: torch.Tensor, off: torch.Tensor, topk_w: torch.Tensor, w2: torch.Tensor,
                w2_s: torch.Tensor, slots: int, h_eps: float, h_smin: float,
                pow2: float) -> torch.Tensor:
    """y f32 [P, H] = topk_w * (quant(h) . w2[e]) for every local pair;
    (h_eps, h_smin) = fp8_moe.quant_params(ue8m0, "h")."""
    idim = hl.specialize(h.size(1))
    e_cnt = hl.specialize(w2.size(0))
    hdim = hl.specialize(w2.size(1))
    pairs = pair_list.size(0)
    tw = topk_w.reshape(pairs)
    w2r = w2.view(e_cnt * hdim, idim)
    y = torch.empty([pairs, hdim], dtype=torch.float32, device=h.device)
    bn = hl.register_block_size(16, 128)
    br = hl.register_block_size(16, 64)
    for tile_s, tile_h in hl.tile([slots, hdim], block_size=[1, bn]):
        e = slot_expert[tile_s.begin]
        if e >= 0:
            start = off[e]
            n = cnt[e]
            h_rows = e * hdim + tile_h.index
            for tile_r in hl.tile(n, block_size=br):
                pr = pair_list[start + tile_r.index]
                acc = hl.zeros([tile_r, tile_h], dtype=torch.float32)
                for tile_k in hl.tile(idim, block_size=128):
                    hv = h[pr, tile_k]
                    sh = torch.clamp_min(
                        torch.clamp_min(torch.amax(torch.abs(hv), dim=1), h_eps) / 448.0, h_smin)
                    p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(sh, 1e-10))))
                    p2 = torch.where(p2 < sh, p2 * 2.0, p2)
                    p2 = torch.where(p2 * 0.5 >= sh, p2 * 0.5, p2)
                    sh = sh + pow2 * (p2 - sh)
                    hq = torch.clamp(hv / sh[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
                    pd = hl.dot(hq, w2r[h_rows, tile_k].T, out_dtype=torch.float32)
                    acc = acc + pd * sh[:, None] * w2_s[e, tile_h.begin // 128, tile_k.id]
                y[pr, tile_h] = acc * tw[pr].to(torch.float32)[:, None]
    return y


def fp8_moe_combine(y: torch.Tensor, topk_ids: torch.Tensor, out: torch.Tensor, id_base: int,
                    e_count: int) -> torch.Tensor:
    """out bf16 [M, H] = sum over local k ascending of y[t*topk + k] (f32)."""
    m = out.size(0)
    hdim = hl.specialize(out.size(1))
    topk = hl.specialize(topk_ids.size(1))
    ids = topk_ids.reshape(m * topk)
    bt = hl.register_block_size(1, 16)
    bo = hl.register_block_size(64, 1024)
    for tile_t, tile_o in hl.tile([m, hdim], block_size=[bt, bo]):
        tot = hl.zeros([tile_t, tile_o], dtype=torch.float32)
        for k in hl.static_range(topk):
            p = tile_t.index * topk + k
            lid = ids[p].to(torch.int32) - id_base
            ok = (lid >= 0) & (lid < e_count)
            tot = torch.where(ok[:, None], tot + y[p, tile_o], tot)
        out[tile_t, tile_o] = tot.to(torch.bfloat16)
    return out


def fp8_moe_fc2_combine(h: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                        w2: torch.Tensor, w2_s: torch.Tensor, out: torch.Tensor, id_base: int,
                        h_eps: float, h_smin: float, pow2: float) -> torch.Tensor:
    """Token-major fc2 + combine: out[t, tile] = bf16(sum_k local tw * (quant(h[p])
    . w2[e, tile])), k ascending (exact e4m3 products, f32 sums)."""
    m = out.size(0)
    hdim = hl.specialize(out.size(1))
    topk = hl.specialize(topk_ids.size(1))
    e_cnt = hl.specialize(w2.size(0))
    idim = hl.specialize(w2.size(2))
    pairs = m * topk
    ids = topk_ids.reshape(pairs)
    tw = topk_w.reshape(pairs)
    w2r = w2.view(e_cnt * hdim, idim)
    bo = hl.register_block_size(16, 128)
    for tile_t, tile_o in hl.tile([m, hdim], block_size=[1, bo]):
        t = tile_t.begin
        tot = hl.zeros([tile_o], dtype=torch.float32)
        for k in hl.static_range(topk):
            p = t * topk + k
            lid = ids[p].to(torch.int32) - id_base
            ok = (lid >= 0) & (lid < e_cnt)
            # ponytail: an off-rank pair streams expert 0's rows and is dropped
            # by the where (EP only; TP layers never have one)
            le = torch.where(ok, lid, 0)
            rows = le * hdim + tile_o.index
            acc = hl.zeros([tile_o], dtype=torch.float32)
            for tile_k in hl.tile(idim, block_size=128):
                hv = h[p, tile_k]
                sh = torch.clamp_min(
                    torch.clamp_min(torch.amax(torch.abs(hv), dim=0), h_eps) / 448.0, h_smin)
                p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(sh, 1e-10))))
                p2 = torch.where(p2 < sh, p2 * 2.0, p2)
                p2 = torch.where(p2 * 0.5 >= sh, p2 * 0.5, p2)
                sh = sh + pow2 * (p2 - sh)
                hq = torch.clamp(hv / sh, -448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)
                w = w2r[rows, tile_k].to(torch.float32)
                blk = torch.sum(w * hq[None, :], dim=1)
                acc = acc + blk * sh * w2_s[le, tile_o.begin // 128, tile_k.id]
            tot = torch.where(ok, tot + acc * tw[p].to(torch.float32), tot)
        out[t, tile_o] = tot.to(torch.bfloat16)
    return out


BODIES = {ROUTE_QUANT: fp8_moe_route_quant, FC1: fp8_moe_fc1, FC2: fp8_moe_fc2,
          COMBINE: fp8_moe_combine, FC2_COMBINE: fp8_moe_fc2_combine}


def run_plan(kerns, plan: str, x, topk_ids, topk_w, w13, w13_s, w2, w2_s, out, ue8m0: bool,
             id_base: int):
    """One MoE forward on `kerns` (PLANS[plan] order, Helion kernels or the
    bodies under interpret mode) -> out."""
    e = w13.shape[0]
    pow2 = 1.0 if ue8m0 else 0.0
    h_eps, h_smin = (float(v) for v in fm.quant_params(bool(ue8m0), "h"))
    slots = min(e, topk_ids.numel())
    xq, xs, pl, se, cnt, off = kerns[0](x, topk_ids, e, int(id_base), pow2)
    h = kerns[1](xq, xs, pl, se, cnt, off, w13, w13_s, topk_ids.shape[1], slots)
    if plan == "fused3":
        return kerns[2](h, topk_ids, topk_w, w2, w2_s, out, int(id_base), h_eps, h_smin, pow2)
    y = kerns[2](h, pl, se, cnt, off, topk_w, w2, w2_s, slots, h_eps, h_smin, pow2)
    return kerns[3](y, topk_ids, out, int(id_base), e)


# ---------------------------------------------------------------------------
# serving: drop-in for _native.fp8_moe_cuda, per-M plan + bucket choice
# ---------------------------------------------------------------------------
def layer_case(cfg: dict, numtokens: int) -> dict:
    """The lab case (configs/helion/<kernel>/<platform>.json key) for a layer."""
    return {"experts": cfg["E"], "hidden": cfg["H"], "intermediate": cfg["I"],
            "topk": cfg["topk"], "numtokens": numtokens}


def plan_for(m: int, fused: bool) -> str:
    return "fused3" if nm.launch_mode(m, fused) == 2 else "front4"


class ServingOp:
    """Same call signature as _native.fp8_moe_cuda. Per M <= max_m: the
    plan's Helion kernels when every one is tuned for bucket(M), else (auto)
    the oxide op with the identical arguments. Workspace, mode and stream_ptr
    are oxide-only: Helion allocates its scratch (graph pool under capture)
    and launches on torch's current stream (= the stream the lane passes)."""

    def __init__(self, cfg: dict, mode: str):
        self.plans, gaps = {}, {}
        for m in range(1, cfg["max_m"] + 1):
            b, plan = hk.bucket(m, BUCKETS), plan_for(m, cfg.get("fused", True))
            case = layer_case(cfg, b)
            lack = [k for k in PLANS[plan] if hk.config(k, case) is None]
            if lack:
                gaps.setdefault((b, plan), lack)
            else:
                self.plans[m] = (b, plan, [hk.compiled(BODIES[k], k, case) for k in PLANS[plan]])
        if mode == "helion" and gaps:
            raise RuntimeError(
                f"{MARKER} {fm.HELION_ENV}=helion but no tuned config on {hk.platform()} for "
                f"E={cfg['E']} H={cfg['H']} I={cfg['I']} topk={cfg['topk']}: "
                + "; ".join(f"numtokens {b} {p}: {', '.join(k)}" for (b, p), k in sorted(gaps.items()))
                + f" (tune them in the autotune lab, or {fm.HELION_ENV}=auto)")
        self.native = None
        if gaps:
            from suffix_hybrid import oxide_kernels
            self.native = oxide_kernels.native()
        # one M per (bucket, plan) compiles it before CUDA-graph capture; M=1
        # too (Helion's dynamic shapes specialize size 1)
        warm = {}
        for m, (b, plan, _) in self.plans.items():
            warm[(b, plan)] = max(m, warm.get((b, plan), 0))
        self.warm_ms = sorted(set(warm.values()) | ({1} if 1 in self.plans else set()))
        served = sorted({(b, p) for b, p, _ in self.plans.values()})
        self.summary = (f"helion {served or '-'}, oxide {sorted(gaps) or '-'} "
                        f"({fm.HELION_ENV}={mode}, {hk.platform()})")

    def kernel(self, m: int):
        return self.plans.get(m)

    def __call__(self, x, topk_ids, topk_w, w13, w13_s, w2, w2_s, ws_aq, ws_as, ws_inter, ws_hq,
                 ws_hs, ws_y, ws_route, out, ue8m0, id_base, mode, stream_ptr):
        entry = self.plans.get(x.shape[0])
        if entry is None:
            return self.native.fp8_moe_cuda(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, ws_aq, ws_as,
                                            ws_inter, ws_hq, ws_hs, ws_y, ws_route, out, ue8m0,
                                            id_base, mode, stream_ptr)
        run_plan(entry[2], entry[1], x, topk_ids, topk_w, w13, w13_s, w2, w2_s, out, ue8m0,
                 id_base)
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


def _cases_for(plan_kernel: str):
    """A kernel is tuned only at the buckets whose plan launches it (lane default fused)."""
    def pick(model):
        return [c for c in cases(model) if plan_kernel in PLANS[plan_for(c["numtokens"], True)]]
    return pick


def route_ref(topk_ids, e_count: int, id_base: int):
    """Torch twin of the route root: (pair_list, slot_expert, cnt, off) int32."""
    pairs = topk_ids.numel()
    dev = topk_ids.device
    loc = topk_ids.reshape(-1).long() - id_base
    key = torch.where((loc >= 0) & (loc < e_count), loc, torch.full_like(loc, e_count))
    order = torch.argsort(key * pairs + torch.arange(pairs, device=dev))
    cnt = torch.bincount(key, minlength=e_count + 1)[:e_count]
    off = torch.cumsum(cnt, 0) - cnt
    act = torch.nonzero(cnt > 0).reshape(-1)
    slot = torch.full((e_count,), -1, dtype=torch.long, device=dev)
    slot[:act.numel()] = act
    return order.int(), slot.int(), cnt.int(), off.int()


def _device():
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() \
        else torch.device("cpu")


@functools.lru_cache(maxsize=1)
def _layer(e, hidden, inter, dev):
    """One layer's weights per case, shared by every rotating copy: the lab
    makes up to 64 copies to spill L2 and a fresh ~600 MB expert set per
    seed ran a 32 GB RTX 5090 out of memory (lfm2.5, run
    20261017T160531Z-rtx-5090-2d59). The weights already exceed L2, so the
    copies only need their own x / routing."""
    return fm.make_problem(e, hidden, inter, device=dev, seed=0)


def _problem(case, seed):
    """(p, x, ids, tw): fp8_moe.make_problem's layer (vLLM Fp8MoEMethod
    layout, per-block amax scales) + uniform distinct top-k routing."""
    dev = _device()
    m, e, k = case["numtokens"], case["experts"], case["topk"]
    p = _layer(e, case["hidden"], case["intermediate"], dev)
    ids, tw = nm.rand_routing(m, e, k, dev, seed=1000 + seed)
    gen = torch.Generator(device="cpu").manual_seed(2000 + seed)
    x = torch.randn(m, case["hidden"], generator=gen).to(dev, torch.bfloat16)
    return p, x, ids, tw


def _xq_ref(x, ue8m0=False):
    q, s = fm.quant_fp8(x.float(), ue8m0)
    return q.view(torch.float8_e4m3fn), s


def _h_ref(p, xq, xs, ids):
    """(h f32 [P, I] = silu(gate) * up from the quantized x, f64 math; local rows)."""
    m, topk = ids.shape
    e_count, two_i = p["w13"].shape[:2]
    i = two_i // 2
    xd = fm.dequant(xq, xs).double()
    h = torch.zeros(m * topk, i, dtype=torch.float64, device=xq.device)
    valid = torch.zeros(m * topk, dtype=torch.bool, device=xq.device)
    for e, pr in nm._groups(ids, e_count).items():
        gu = xd[pr // topk] @ fm.expert_w(p["w13"], p["w13_s"], e).double().T
        h[pr] = fm.silu(gu[:, :i]) * gu[:, i:]
        valid[pr] = True
    return h.float(), valid


def _y_ref(h, ids, tw, w2, w2_s, ue8m0=False):
    """(y f64 [P, H] = tw * (quant_h(h) . w2[e]); local rows)."""
    e_count, hdim = w2.shape[:2]
    hd = fm.qdq(h, ue8m0, "h").double()
    y = torch.zeros(h.shape[0], hdim, dtype=torch.float64, device=h.device)
    valid = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
    for e, pr in nm._groups(ids, e_count).items():
        y[pr] = (hd[pr] @ fm.expert_w(w2, w2_s, e).double().T) * tw.reshape(-1)[pr, None].double()
        valid[pr] = True
    return y, valid


def _stage_inputs(case, seed):
    p, x, ids, tw = _problem(case, seed)
    xq, xs = _xq_ref(x)
    pl, se, cnt, off = route_ref(ids, p["w13"].shape[0], 0)
    return p, x, ids, tw, xq, xs, pl, se, cnt, off


def _make_route_quant(case, seed):
    p, x, ids, _ = _problem(case, seed)
    return x, ids, p["w13"].shape[0], 0, 0.0


def _make_fc1(case, seed):
    p, _, ids, _, xq, xs, pl, se, cnt, off = _stage_inputs(case, seed)
    return xq, xs, pl, se, cnt, off, p["w13"], p["w13_s"], ids.shape[1], min(se.numel(), pl.numel())


def _make_fc2(case, seed):
    p, _, ids, tw, xq, xs, pl, se, cnt, off = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xq, xs, ids)
    eps, smin = fm.quant_params(False, "h")
    return (h, pl, se, cnt, off, tw, p["w2"], p["w2_s"], min(se.numel(), pl.numel()), float(eps),
            float(smin), 0.0)


def _make_combine(case, seed):
    p, x, ids, _ = _problem(case, seed)
    gen = torch.Generator(device="cpu").manual_seed(3000 + seed)
    y = torch.randn(ids.numel(), case["hidden"], generator=gen).to(x.device)
    out = torch.empty(x.shape[0], case["hidden"], dtype=torch.bfloat16, device=x.device)
    return y, ids, out, 0, p["w13"].shape[0]


def _make_fc2_combine(case, seed):
    p, x, ids, tw, xq, xs, *_ = _stage_inputs(case, seed)
    h, _ = _h_ref(p, xq, xs, ids)
    eps, smin = fm.quant_params(False, "h")
    out = torch.empty(x.shape[0], case["hidden"], dtype=torch.bfloat16, device=x.device)
    return h, ids, tw, p["w2"], p["w2_s"], out, 0, float(eps), float(smin), 0.0


def _ref_route_quant(x, topk_ids, e_count, id_base, pow2):
    return (*_xq_ref(x, pow2 > 0.5), *route_ref(topk_ids, e_count, id_base))


def _ids_from_route(pl, cnt, e_count, topk):
    """Local ids [M, K] back from (pair_list, cnt): off-rank pairs -> -1."""
    key = torch.repeat_interleave(
        torch.arange(e_count + 1, device=pl.device),
        torch.cat([cnt.long(), (pl.numel() - cnt.sum()).reshape(1)]))
    lid = torch.empty(pl.numel(), dtype=torch.long, device=pl.device)
    lid[pl.long()] = torch.where(key < e_count, key, -1)
    return lid.view(-1, topk)


def _ref_fc1(xq, xs, pl, se, cnt, off, w13, w13_s, topk, slots):
    ids = _ids_from_route(pl, cnt, w13.shape[0], topk)
    return _h_ref({"w13": w13, "w13_s": w13_s}, xq, xs, ids)


def _ref_fc2(h, pl, se, cnt, off, tw, w2, w2_s, slots, h_eps, h_smin, pow2):
    ids = _ids_from_route(pl, cnt, w2.shape[0], tw.shape[1])
    return _y_ref(h, ids, tw, w2, w2_s, pow2 > 0.5)


def _ref_combine(y, topk_ids, out, id_base, e_count):
    lid = nm.to_local(topk_ids, id_base, e_count)
    m, k = lid.shape
    yk = y.view(m, k, -1)
    acc = torch.zeros(m, y.shape[1], dtype=torch.float32, device=y.device)
    for j in range(k):  # the kernel's fixed k order, f32
        acc = torch.where((lid[:, j] >= 0)[:, None], acc + yk[:, j], acc)
    return acc.bfloat16()


def _ref_fc2_combine(h, topk_ids, topk_w, w2, w2_s, out, id_base, h_eps, h_smin, pow2):
    lid = nm.to_local(topk_ids, id_base, w2.shape[0])
    y, valid = _y_ref(h, lid, topk_w, w2, w2_s, pow2 > 0.5)
    return (y * valid[:, None]).view(lid.shape[0], lid.shape[1], -1).sum(1)


def _rel(a, b) -> float:
    return nm._rel(a.float(), b)


def _check_route_quant(out, ref):
    xq, xs, *route = out
    xq_r, xs_r, *route_r = ref
    same = [bool(torch.equal(a.int(), b.int())) for a, b in zip(route, route_r, strict=True)]
    r = _rel(fm.dequant(xq, xs), fm.dequant(xq_r, xs_r))
    ok = all(same) and r <= 1e-3
    return ok, (f"route pair_list/slot/cnt/off equal={same}, x quant dequant rel {r:.2e} "
                "(gate 1e-3: 1-ulp div.full scales)")


def _check_rows(tol: float, what: str):
    def check(out, ref):
        want, rows = ref
        r = _rel(out[rows], want[rows]) if bool(rows.any()) else 0.0
        ok = r <= tol and bool(torch.isfinite(out[rows]).all())
        return ok, f"{what} rel vs f64 spec {r:.2e} on {int(rows.sum())} local rows (gate {tol:.0e})"
    return check


def _check_combine(out, ref):
    frac = float((out != ref).float().mean())
    return frac <= fm.STAGE_TOL["comb"], f"combine bf16 mismatch fraction {frac:.1e}"


def _check_end(out, ref):
    r = _rel(out, ref)
    ok = r <= fm.REF_TOL and bool(torch.isfinite(out.float()).all())
    return ok, f"rel vs f64 spec {r:.2e} (gate REF_TOL {fm.REF_TOL:.0e})"


def _oxide_args(case, seed, mode: int, keep: int):
    """The oxide op's arguments (stream excluded) running ONLY launch `keep`
    of `mode`'s plan, workspace pre-filled by one full run so the kept launch
    reads real upstream data. Needs the host's mode-word skip mask (module doc)."""
    from suffix_hybrid import oxide_kernels
    oxide_kernels.ensure_loaded(fm.FAMILY)
    native = oxide_kernels.native()
    p, x, ids, tw = _problem(case, seed)
    ws = fm.workspace(x.device, x.shape[0], ids.shape[1], case["hidden"], case["intermediate"],
                      case["experts"])
    out = torch.empty(x.shape[0], case["hidden"], dtype=torch.bfloat16, device=x.device)
    head = (x, ids, tw, p["w13"], p["w13_s"], p["w2"], p["w2_s"], *ws, out, False, 0)
    native.fp8_moe_cuda(*head, mode, torch.cuda.current_stream().cuda_stream)
    launches = 3 if mode == 2 else 4
    skip = ((1 << launches) - 1) & ~(1 << keep)
    return (*head, mode | (skip << SKIP_SHIFT))


def _oxide_baseline(*args):
    from suffix_hybrid import oxide_kernels
    return oxide_kernels.native().fp8_moe_cuda(*args, torch.cuda.current_stream().cuda_stream)


def _entry(name, make, reference, check, mode, keep, oxide_entries):
    return {"name": name, "body": BODIES[name], "cases": _cases_for(name), "make": make,
            "reference": reference, "check": check, "baseline": _oxide_baseline,
            "make_baseline": lambda case, seed: _oxide_args(case, seed, mode, keep),
            "baseline_name": f"oxide fp8_moe {oxide_entries[0]} (plan {mode} launch {keep} alone)",
            "replaces": [rf"^({'|'.join(oxide_entries)})$"]}


KERNELS = [
    # route+quant also stands for the legacy plan's moe_route + moe_quant_rows
    _entry(ROUTE_QUANT, _make_route_quant, _ref_route_quant, _check_route_quant, 1, 0,
           ("moe_route_quant", "moe_route", "moe_quant_rows")),
    # moe_fc1_quant = fc1 + h quant; legacy moe_fc1 the GEMM alone
    _entry(FC1, _make_fc1, _ref_fc1, _check_rows(fm.STAGE_TOL["fc1"], "h"), 1, 1,
           ("moe_fc1_quant", "moe_fc1")),
    # 1e-3, not STAGE_TOL 1e-4: h is requantized here with div.full scales
    _entry(FC2, _make_fc2, _ref_fc2, _check_rows(1e-3, "y"), 1, 2, ("moe_fc2",)),
    _entry(COMBINE, _make_combine, _ref_combine, _check_combine, 1, 3, ("moe_combine",)),
    _entry(FC2_COMBINE, _make_fc2_combine, _ref_fc2_combine, _check_end, 2, 2,
           ("moe_fc2_combine",)),
]
