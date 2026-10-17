# SPDX-License-Identifier: Apache-2.0
"""Helion port of kernels-oxide/fp8_moe: the FP8 block-scaled (e4m3 weights,
128x128 f32 weight blocks, dynamic 1x128 activation groups) routed-experts
decode MoE (suffix_hybrid/kernels/fp8_moe.py) — one Helion kernel per oxide
launch, chained on the stream like the oxide plans (no grid barrier: a
persistent spin-barrier deadlocks once its grid exceeds co-residency, lab
2026-10-17). Every kernel reads and writes the OXIDE WORKSPACE layout
(fp8_moe.workspace), so any launch can run on either backend:

  fp8_moe_route_quant  (= moe_route_quant) three independent roots:
      route   ws_route = [slot_expert E | slot_off E | slot_cnt E | pair_list P]
              exactly moe_route's: active local experts ascending (then -1,
              cnt 0), pairs ascending per expert; rank counting, no atomics
      quant x vLLM per_token_group_quant_fp8 per (token, 128 cols): eps
              1e-10, s = amax/448 (pow2: UE8M0) -> ws_aq e4m3, ws_as f32
  fp8_moe_fc1          (= moe_fc1_quant) grid (slot, 128-col I group): gate +
      up dots share the A tile, per-128-K-block rescale acc += dot * s_x * s_w
      (Triton order), h = silu(gate) * up, h quantized per (pair, 128 group)
      in-tile (h-stage eps / floor) -> ws_hq e4m3, ws_hs f32
  fp8_moe_fc2          (= moe_fc2) grid (slot, H tile): y = topk_w * acc -> ws_y
  fp8_moe_combine      (= moe_combine) out[t] = bf16(sum_k y[t*topk + k]), local
      k ascending; a token with no local expert is exact 0
  fp8_moe_fc2_combine  (= moe_fc2_combine) token-major GEMV per (token, H
      tile), k ascending (decode M <= FUSED_MAX_M)

Plans (nvfp4_moe.launch_mode per M): fused3 = route_quant, fc1, fc2_combine;
front4 = route_quant, fc1, fc2, combine (the lane's legacy6 maps to front4).

Numerics = oxide's quant bit for bit: every quant division is IEEE div_rn
(Triton's `/` is div.full and Helion lowers even Inductor's div_rn to `/`:
bf16 inputs hit exact e4m3 ties often, so div.full flipped ~1e-2 of the
rounding decisions -> x-quant rel 2e-3 in the lab, 2026-10-17); f32->e4m3 =
Triton's rtne + pre-clamp = cvt.rn.satfinite. Rest = the lane's contract
(f64 spec moe_ref, REF_TOL), f32 accumulation, fixed-order combine.

Tunables (explicit bounds keep configs compiling fast): route tiles 16..128,
x-quant rows 1..16, fc2 N tile 16..128 (inside one 128-row scale block),
expert-row tiles 16..64 (dot rows >= 16), combine tiles, warps, stages, pid
type, indexing / eviction per load. K = the 128 scale block (num_stages
pipelines it); fc1's N = the 128-col h quant group (whole group per program).

Serving: SUFFIX_HELION_FP8_MOE (helion_kernels.backend) inside the enabled
SUFFIX_FP8_MOE lane — fp8_moe.helion_op builds a ServingOp, a drop-in for
_native.fp8_moe_cuda (same signature). auto: per launch, Helion where its own
config is shipped for bucket(M), the oxide launch alone elsewhere (same
workspace); helion: every launch of every served M tuned, else load error.

Per-launch oxide (mixing, the lab baselines): src/fp8_moe_oxide.rs reads the
`mode` word's bits SKIP_SHIFT.. as a skip mask (launch i of the plan skipped
when bit SKIP_SHIFT + i is set, plan = word & 0xFF) and src/lib.rs exports
`_native.FP8_MOE_SKIP_SHIFT = 8`. An older _native without the marker mixes
nothing (an M runs Helion only when its whole plan is tuned).
"""
import functools
import os

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
SKIP_SHIFT = 8  # oxide: launch i of the plan skipped when mode bit 8+i is set (host change)
# Helion's eager reference mode runs the bodies as torch: IEEE `/` there,
# hl.inline_triton (no eager form) only in compiled code
_EAGER = os.environ.get("HELION_INTERPRET", "") == "1"
ROUTE_QUANT, FC1, FC2, COMBINE, FC2_COMBINE = (
    "fp8_moe_route_quant", "fp8_moe_fc1", "fp8_moe_fc2", "fp8_moe_combine",
    "fp8_moe_fc2_combine")
PLANS = {"front4": (ROUTE_QUANT, FC1, FC2, COMBINE), "fused3": (ROUTE_QUANT, FC1, FC2_COMBINE)}
OXIDE_MODE = {"front4": 1, "fused3": 2}


# ---------------------------------------------------------------------------
# Helion bodies (module level, undecorated: the lab and compiled() wrap them;
# literals only inside, they are traced). Outputs are written in place.
# ---------------------------------------------------------------------------
def fp8_moe_route_quant(x: torch.Tensor, topk_ids: torch.Tensor, aq: torch.Tensor,
                        a_s: torch.Tensor, slot_expert: torch.Tensor, slot_off: torch.Tensor,
                        slot_cnt: torch.Tensor, pair_list: torch.Tensor, e_count: int,
                        id_base: int, pow2: float):
    """x bf16 [M, H], topk_ids int [M, K] GLOBAL -> aq e4m3 [M, H], a_s f32
    [M, H/128] and the four i32 sections of ws_route (moe_route layout, one
    view each: two roots writing one tensor is a cross-loop dependency to
    Helion); E = e_count <= 256 local experts from id_base; pow2 1.0 = UE8M0."""
    m = x.size(0)
    hdim = hl.specialize(x.size(1))
    topk = hl.specialize(topk_ids.size(1))
    pairs = m * topk
    ids = topk_ids.reshape(pairs)
    bp = hl.register_block_size(16, 128)
    bq = hl.register_block_size(16, 128)
    bq2 = hl.register_block_size(16, 64)
    bt = hl.register_block_size(1, 16)

    # route, ONE root (the four sections are views of one workspace buffer:
    # Helion orders roots that share storage). pair_list: sorted position =
    # #pairs before it in (key, p) order; key = local id, off-rank pairs = E
    # (sorted last: listed, never in a segment)
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
        # the slot lists: the first program, one pass over all E (<= 256)
        if tile_p.begin == 0:
            for tile_e in hl.tile(e_count, block_size=256):
                c = hl.zeros([tile_e], dtype=torch.int32)
                lt = hl.zeros([tile_e], dtype=torch.int32)
                for tile_q2 in hl.tile(pairs, block_size=bq2):
                    kq2 = ids[tile_q2].to(torch.int32) - id_base
                    kq2 = torch.where((kq2 >= 0) & (kq2 < e_count), kq2, e_count)
                    vq = (tile_q2.index < pairs)[None, :]
                    c = c + torch.sum(((kq2[None, :] == tile_e.index[:, None]) & vq).to(
                        torch.int32), dim=1).to(torch.int32)
                    lt = lt + torch.sum(((kq2[None, :] < tile_e.index[:, None]) & vq).to(
                        torch.int32), dim=1).to(torch.int32)
                ev = tile_e.index
                c = torch.where(ev < e_count, c, 0)
                act = (c > 0).to(torch.int32)
                rank = torch.cumsum(act, 0) - act
                n_act = torch.sum(act)
                # active expert -> slot rank; inactive -> the tail (slot_expert -1, cnt 0)
                dst = torch.where(c > 0, rank, n_act + ev - rank)
                slot_expert[dst] = torch.where(c > 0, ev, -1)
                slot_off[dst] = lt
                slot_cnt[dst] = c
    # x quant per (token, 128-col group), IEEE divisions like the CUDA quant
    for tile_t, tile_g in hl.tile([m, hdim], block_size=[bt, 128]):
        xv = x[tile_t, tile_g].to(torch.float32)
        amax = torch.clamp_min(torch.amax(torch.abs(xv), dim=1), 1e-10)
        if _EAGER:
            s = amax / torch.full_like(amax, 448.0)
        else:
            s = hl.inline_triton("tl.div_rn({0}, {1})",
                                 args=(amax, torch.zeros_like(amax) + 448.0), output_like=amax)
        p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(s, 1e-10))))
        p2 = torch.where(p2 < s, p2 * 2.0, p2)
        p2 = torch.where(p2 * 0.5 >= s, p2 * 0.5, p2)
        s = s + pow2 * (p2 - s)  # exact blend (Sterbenz: s <= p2 <= 2s)
        sb = torch.zeros_like(xv) + s[:, None]
        if _EAGER:
            qv = xv / sb
        else:
            qv = hl.inline_triton("tl.div_rn({0}, {1})", args=(xv, sb), output_like=xv)
        aq[tile_t, tile_g] = torch.clamp(qv, -448.0, 448.0).to(torch.float8_e4m3fn)
        a_s[tile_t, tile_g.id] = s
    return aq, a_s, slot_expert, slot_off, slot_cnt, pair_list


def fp8_moe_fc1(aq: torch.Tensor, a_s: torch.Tensor, route: torch.Tensor, w13: torch.Tensor,
                w13_s: torch.Tensor, hq: torch.Tensor, hs: torch.Tensor, topk: int, slots: int,
                h_eps: float, h_smin: float, pow2: float):
    """hq e4m3 [P, I] + hs f32 [P, I/128] = h-stage quant of silu(gate) * up
    for every local pair (off-rank rows unwritten); w13 e4m3 [E, 2I, H] =
    [gate; up], w13_s f32 [E, 2I/128, H/128]; slots = min(E, P);
    (h_eps, h_smin) = fp8_moe.quant_params(ue8m0, "h")."""
    hdim = hl.specialize(aq.size(1))
    e_cnt = hl.specialize(w13.size(0))
    idim = hl.specialize(w13.size(1)) // 2
    # weights as flat rows: an expert's tile is the row gather e*rows + tile.index
    # (a 0-d data-dependent index next to tile dims does not lower)
    w13r = w13.view(e_cnt * 2 * idim, hdim)
    s13v = w13_s.view(e_cnt, 2, idim // 128, hdim // 128)
    br = hl.register_block_size(16, 64)
    for tile_s, tile_i in hl.tile([slots, idim], block_size=[1, 128]):
        s = tile_s.begin
        e = route[s]
        if e >= 0:
            start = route[e_cnt + s]
            n = route[2 * e_cnt + s]
            g_rows = e * (2 * idim) + tile_i.index
            for tile_r in hl.tile(n, block_size=br):
                pr = route[3 * e_cnt + start + tile_r.index]
                tok = pr // topk
                gate = hl.zeros([tile_r, tile_i], dtype=torch.float32)
                up = hl.zeros([tile_r, tile_i], dtype=torch.float32)
                for tile_k in hl.tile(hdim, block_size=128):
                    a = aq[tok, tile_k]
                    sa = a_s[tok, tile_k.id]
                    pg = hl.dot(a, w13r[g_rows, tile_k].T, out_dtype=torch.float32)
                    pu = hl.dot(a, w13r[g_rows + idim, tile_k].T, out_dtype=torch.float32)
                    gate = gate + pg * sa[:, None] * s13v[e, 0, tile_i.id, tile_k.id]
                    up = up + pu * sa[:, None] * s13v[e, 1, tile_i.id, tile_k.id]
                h = gate * torch.sigmoid(gate) * up
                amax = torch.clamp_min(torch.amax(torch.abs(h), dim=1), h_eps)
                if _EAGER:
                    sh = amax / torch.full_like(amax, 448.0)
                else:
                    sh = hl.inline_triton("tl.div_rn({0}, {1})",
                                          args=(amax, torch.zeros_like(amax) + 448.0),
                                          output_like=amax)
                sh = torch.clamp_min(sh, h_smin)
                p2 = torch.exp2(torch.ceil(torch.log2(torch.clamp_min(sh, 1e-10))))
                p2 = torch.where(p2 < sh, p2 * 2.0, p2)
                p2 = torch.where(p2 * 0.5 >= sh, p2 * 0.5, p2)
                sh = sh + pow2 * (p2 - sh)
                sb = torch.zeros_like(h) + sh[:, None]
                if _EAGER:
                    qv = h / sb
                else:
                    qv = hl.inline_triton("tl.div_rn({0}, {1})", args=(h, sb), output_like=h)
                hq[pr, tile_i] = torch.clamp(qv, -448.0, 448.0).to(torch.float8_e4m3fn)
                hs[pr, tile_i.id] = sh
    return hq, hs


def fp8_moe_fc2(hq: torch.Tensor, hs: torch.Tensor, route: torch.Tensor, topk_w: torch.Tensor,
                w2: torch.Tensor, w2_s: torch.Tensor, y: torch.Tensor, slots: int) -> torch.Tensor:
    """y f32 [P, H] = topk_w * (hq . w2[e]) block-rescaled, every local pair."""
    idim = hl.specialize(hq.size(1))
    e_cnt = hl.specialize(w2.size(0))
    hdim = hl.specialize(w2.size(1))
    tw = topk_w.reshape(hq.size(0))
    w2r = w2.view(e_cnt * hdim, idim)
    bn = hl.register_block_size(16, 128)
    br = hl.register_block_size(16, 64)
    for tile_s, tile_h in hl.tile([slots, hdim], block_size=[1, bn]):
        s = tile_s.begin
        e = route[s]
        if e >= 0:
            start = route[e_cnt + s]
            n = route[2 * e_cnt + s]
            h_rows = e * hdim + tile_h.index
            for tile_r in hl.tile(n, block_size=br):
                pr = route[3 * e_cnt + start + tile_r.index]
                acc = hl.zeros([tile_r, tile_h], dtype=torch.float32)
                for tile_k in hl.tile(idim, block_size=128):
                    pd = hl.dot(hq[pr, tile_k], w2r[h_rows, tile_k].T, out_dtype=torch.float32)
                    acc = acc + pd * hs[pr, tile_k.id][:, None] * w2_s[e, tile_h.begin // 128,
                                                                       tile_k.id]
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


def fp8_moe_fc2_combine(hq: torch.Tensor, hs: torch.Tensor, topk_ids: torch.Tensor,
                        topk_w: torch.Tensor, w2: torch.Tensor, w2_s: torch.Tensor,
                        out: torch.Tensor, id_base: int) -> torch.Tensor:
    """Token-major fc2 + combine: out[t, tile] = bf16(sum_k local tw * (hq[p]
    . w2[e, tile]) block-rescaled), k ascending (exact e4m3 products, f32 sums)."""
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
                a = hq[p, tile_k].to(torch.float32)
                w = w2r[rows, tile_k].to(torch.float32)
                blk = torch.sum(w * a[None, :], dim=1)
                acc = acc + blk * hs[p, tile_k.id] * w2_s[le, tile_o.begin // 128, tile_k.id]
            tot = torch.where(ok, tot + acc * tw[p].to(torch.float32), tot)
        out[t, tile_o] = tot.to(torch.bfloat16)
    return out


BODIES = {ROUTE_QUANT: fp8_moe_route_quant, FC1: fp8_moe_fc1, FC2: fp8_moe_fc2,
          COMBINE: fp8_moe_combine, FC2_COMBINE: fp8_moe_fc2_combine}


def ws_views(ws, m: int, topk: int, hdim: int, idim: int, e_count: int):
    """(aq, a_s, hq, hs, y, route) views of the oxide workspace (fp8_moe.workspace)."""
    p = m * topk
    f8 = torch.float8_e4m3fn
    return (ws[0][:m * hdim].view(f8).view(m, hdim), ws[1][:m * hdim // 128].view(m, -1),
            ws[3][:p * idim].view(f8).view(p, idim), ws[4][:p * idim // 128].view(p, -1),
            ws[5][:p * hdim].view(p, hdim), ws[6][:3 * e_count + p])


def run_launch(name: str, kern, x, topk_ids, topk_w, w13, w13_s, w2, w2_s, views, out,
               ue8m0: bool, id_base: int):
    """One launch of the plan on Helion (`kern` = compiled kernel or, under
    interpret mode, the body wrapped by helion.kernel) over the oxide workspace."""
    aq, a_s, hq, hs, y, route = views
    e = w13.shape[0]
    pow2 = 1.0 if ue8m0 else 0.0
    slots = min(e, topk_ids.numel())
    if name == ROUTE_QUANT:
        kern(x, topk_ids, aq, a_s, route[:e], route[e:2 * e], route[2 * e:3 * e], route[3 * e:],
             e, int(id_base), pow2)
    elif name == FC1:
        h_eps, h_smin = (float(v) for v in fm.quant_params(bool(ue8m0), "h"))
        kern(aq, a_s, route, w13, w13_s, hq, hs, topk_ids.shape[1], slots, h_eps, h_smin, pow2)
    elif name == FC2:
        kern(hq, hs, route, topk_w, w2, w2_s, y, slots)
    elif name == COMBINE:
        kern(y, topk_ids, out, int(id_base), e)
    else:
        kern(hq, hs, topk_ids, topk_w, w2, w2_s, out, int(id_base))


# ---------------------------------------------------------------------------
# serving: drop-in for _native.fp8_moe_cuda, per-M plan, per-launch backend
# ---------------------------------------------------------------------------
def layer_case(cfg: dict, numtokens: int) -> dict:
    """The lab case (configs/helion/<kernel>/<platform>.json key) for a layer."""
    return {"experts": cfg["E"], "hidden": cfg["H"], "intermediate": cfg["I"],
            "topk": cfg["topk"], "numtokens": numtokens}


def plan_for(m: int, fused: bool) -> str:
    return "fused3" if nm.launch_mode(m, fused) == 2 else "front4"


def host_skips(native) -> bool:
    """The host op runs single launches (mode-word skip mask, module doc)."""
    return getattr(native, "FP8_MOE_SKIP_SHIFT", None) == SKIP_SHIFT


class ServingOp:
    """Same call signature as _native.fp8_moe_cuda. Per M <= max_m: the plan
    of launch_mode(M); each launch on Helion where its config is shipped for
    bucket(M), else the oxide launch alone over the same workspace (bit-equal
    interfaces). A host without skip masks mixes nothing: an M runs Helion
    only when its whole plan is tuned. Helion launches run on torch's current
    stream (= the lane's stream_ptr)."""

    def __init__(self, cfg: dict, mode: str, native=None):
        self.dims = (cfg["topk"], cfg["H"], cfg["I"], cfg["E"])
        self.native = native
        mix = native is not None and host_skips(native)
        self.plans, gaps = {}, {}
        for m in range(1, cfg["max_m"] + 1):
            b, plan = hk.bucket(m, BUCKETS), plan_for(m, cfg.get("fused", True))
            case = layer_case(cfg, b)
            lack = [k for k in PLANS[plan] if hk.config(k, case) is None]
            if lack:
                gaps.setdefault((b, plan), lack)
            if len(lack) == len(PLANS[plan]) or (lack and not mix):
                continue
            self.plans[m] = (b, plan, [None if k in lack else hk.compiled(BODIES[k], k, case)
                                       for k in PLANS[plan]])
        if mode == "helion" and gaps:
            raise RuntimeError(
                f"{MARKER} {fm.HELION_ENV}=helion but no tuned config on {hk.platform()} for "
                f"E={cfg['E']} H={cfg['H']} I={cfg['I']} topk={cfg['topk']}: "
                + "; ".join(f"numtokens {b} {p}: {', '.join(k)}"
                            for (b, p), k in sorted(gaps.items()))
                + f" (tune them in the autotune lab, or {fm.HELION_ENV}=auto)")
        # one M per (bucket, plan) compiles it before CUDA-graph capture; M=1
        # too (Helion's dynamic shapes specialize size 1)
        warm = {}
        for m, (b, plan, _) in self.plans.items():
            warm[(b, plan)] = max(m, warm.get((b, plan), 0))
        self.warm_ms = sorted(set(warm.values()) | ({1} if 1 in self.plans else set()))
        on = sorted({(b, p, tuple(k for k, kern in zip(PLANS[p], ks, strict=True) if kern))
                     for b, p, ks in self.plans.values()})
        self.summary = (f"helion {on or '-'}, oxide {sorted(gaps.items()) or '-'} "
                        f"(mixing {'on' if mix else 'off: host has no skip mask'}, "
                        f"{fm.HELION_ENV}={mode}, {hk.platform()})")

    def kernel(self, m: int):
        return self.plans.get(m)

    def __call__(self, x, topk_ids, topk_w, w13, w13_s, w2, w2_s, ws_aq, ws_as, ws_inter, ws_hq,
                 ws_hs, ws_y, ws_route, out, ue8m0, id_base, mode, stream_ptr):
        ws = (ws_aq, ws_as, ws_inter, ws_hq, ws_hs, ws_y, ws_route)
        entry = self.plans.get(x.shape[0])
        if entry is None:
            return self.native.fp8_moe_cuda(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, *ws, out,
                                            ue8m0, id_base, mode, stream_ptr)
        _, plan, kerns = entry
        views = ws_views(ws, x.shape[0], *self.dims)
        n = len(kerns)
        i = 0
        while i < n:
            if kerns[i] is not None:
                run_launch(PLANS[plan][i], kerns[i], x, topk_ids, topk_w, w13, w13_s, w2, w2_s,
                           views, out, ue8m0, id_base)
                i += 1
                continue
            j = i
            while j < n and kerns[j] is None:
                j += 1
            keep = ((1 << j) - 1) & ~((1 << i) - 1)  # oxide launches [i, j) in one call
            word = OXIDE_MODE[plan] | ((((1 << n) - 1) & ~keep) << SKIP_SHIFT)
            self.native.fp8_moe_cuda(x, topk_ids, topk_w, w13, w13_s, w2, w2_s, *ws, out, ue8m0,
                                     id_base, word, stream_ptr)
            i = j
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
    """Torch twin of moe_route in the workspace layout: int32 [3E + P] =
    slot_expert (active ascending, then -1) | slot_off | slot_cnt (0 past the
    active slots) | pair_list (local pairs by (expert, pair), then off-rank)."""
    pairs = topk_ids.numel()
    dev = topk_ids.device
    loc = topk_ids.reshape(-1).long() - id_base
    key = torch.where((loc >= 0) & (loc < e_count), loc, torch.full_like(loc, e_count))
    order = torch.argsort(key * pairs + torch.arange(pairs, device=dev))
    cnt = torch.bincount(key, minlength=e_count + 1)[:e_count]
    off = torch.cumsum(cnt, 0) - cnt
    act = torch.nonzero(cnt > 0).reshape(-1)
    r = torch.zeros(3 * e_count + pairs, dtype=torch.long, device=dev)
    r[:e_count] = -1
    r[:act.numel()] = act
    r[e_count:e_count + act.numel()] = off[act]
    r[2 * e_count:2 * e_count + act.numel()] = cnt[act]
    r[3 * e_count:] = order
    return r.int()


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


def _local_ids(route, e_count: int, topk: int):
    """Local ids [M, K] back from a route (pair_list + slot lists): off-rank -1."""
    pairs = route.numel() - 3 * e_count
    lid = torch.full((pairs,), -1, dtype=torch.long, device=route.device)
    for s in range(e_count):
        e, o, c = (int(route[s + j * e_count]) for j in range(3))
        if e < 0:
            break
        lid[route[3 * e_count + o:3 * e_count + o + c].long()] = e
    return lid.view(-1, topk)


def _h_ref(w13, w13_s, aq, a_s, lid, ue8m0=False):
    """(hq, hs, local rows): the spec's h quant of silu(gate) * up computed in
    f64 from the quantized x."""
    m, topk = lid.shape
    i = w13.shape[1] // 2
    xd = fm.dequant(aq, a_s).double()
    h = torch.zeros(m * topk, i, dtype=torch.float64, device=aq.device)
    valid = torch.zeros(m * topk, dtype=torch.bool, device=aq.device)
    for e, pr in nm._groups(lid, w13.shape[0]).items():
        gu = xd[pr // topk] @ fm.expert_w(w13, w13_s, e).double().T
        h[pr] = fm.silu(gu[:, :i]) * gu[:, i:]
        valid[pr] = True
    q, s = fm.quant_fp8(h.float(), ue8m0, "h")
    return q.view(torch.float8_e4m3fn), s, valid


def _y_ref(hq, hs, lid, tw, w2, w2_s):
    """(y f64 [P, H] = tw * (hq . w2[e]) block-rescaled; local rows)."""
    hd = fm.dequant(hq, hs).double()
    y = torch.zeros(hq.shape[0], w2.shape[1], dtype=torch.float64, device=hq.device)
    valid = torch.zeros(hq.shape[0], dtype=torch.bool, device=hq.device)
    for e, pr in nm._groups(lid, w2.shape[0]).items():
        y[pr] = (hd[pr] @ fm.expert_w(w2, w2_s, e).double().T) * tw.reshape(-1)[pr, None].double()
        valid[pr] = True
    return y, valid


def _stage_inputs(case, seed):
    p, x, ids, tw = _problem(case, seed)
    e = p["w13"].shape[0]
    aq, a_s = _xq_ref(x)
    route = route_ref(ids, e, 0)
    hq, hs, _ = _h_ref(p["w13"], p["w13_s"], aq, a_s, nm.to_local(ids, 0, e))
    return p, x, ids, tw, aq, a_s, route, hq, hs


def _make_route_quant(case, seed):
    p, x, ids, _ = _problem(case, seed)
    m, h, e = x.shape[0], case["hidden"], case["experts"]
    aq = torch.empty(m, h, dtype=torch.float8_e4m3fn, device=x.device)
    a_s = torch.empty(m, h // 128, dtype=torch.float32, device=x.device)
    route = torch.empty(3 * e + ids.numel(), dtype=torch.int32, device=x.device)
    return x, ids, aq, a_s, route[:e], route[e:2 * e], route[2 * e:3 * e], route[3 * e:], e, 0, 0.0


def _make_fc1(case, seed):
    p, x, ids, _, aq, a_s, route, hq, hs = _stage_inputs(case, seed)
    eps, smin = fm.quant_params(False, "h")
    return (aq, a_s, route, p["w13"], p["w13_s"], torch.empty_like(hq), torch.empty_like(hs),
            ids.shape[1], min(case["experts"], ids.numel()), float(eps), float(smin), 0.0)


def _make_fc2(case, seed):
    p, _, ids, tw, _, _, route, hq, hs = _stage_inputs(case, seed)
    y = torch.empty(hq.shape[0], case["hidden"], dtype=torch.float32, device=hq.device)
    return hq, hs, route, tw, p["w2"], p["w2_s"], y, min(case["experts"], ids.numel())


def _make_combine(case, seed):
    p, x, ids, _ = _problem(case, seed)
    gen = torch.Generator(device="cpu").manual_seed(3000 + seed)
    y = torch.randn(ids.numel(), case["hidden"], generator=gen).to(x.device)
    out = torch.empty(x.shape[0], case["hidden"], dtype=torch.bfloat16, device=x.device)
    return y, ids, out, 0, p["w13"].shape[0]


def _make_fc2_combine(case, seed):
    p, x, ids, tw, _, _, _, hq, hs = _stage_inputs(case, seed)
    out = torch.empty(x.shape[0], case["hidden"], dtype=torch.bfloat16, device=x.device)
    return hq, hs, ids, tw, p["w2"], p["w2_s"], out, 0


def _ref_route_quant(x, topk_ids, aq, a_s, se, so, sc, pl, e_count, id_base, pow2):
    return (*_xq_ref(x, pow2 > 0.5), route_ref(topk_ids, e_count, id_base), e_count)


def _ref_fc1(aq, a_s, route, w13, w13_s, hq, hs, topk, slots, h_eps, h_smin, pow2):
    return _h_ref(w13, w13_s, aq, a_s, _local_ids(route, w13.shape[0], topk), pow2 > 0.5)


def _ref_fc2(hq, hs, route, tw, w2, w2_s, y, slots):
    return _y_ref(hq, hs, _local_ids(route, w2.shape[0], tw.shape[1]), tw, w2, w2_s)


def _ref_combine(y, topk_ids, out, id_base, e_count):
    lid = nm.to_local(topk_ids, id_base, e_count)
    m, k = lid.shape
    yk = y.view(m, k, -1)
    acc = torch.zeros(m, y.shape[1], dtype=torch.float32, device=y.device)
    for j in range(k):  # the kernel's fixed k order, f32
        acc = torch.where((lid[:, j] >= 0)[:, None], acc + yk[:, j], acc)
    return acc.bfloat16()


def _ref_fc2_combine(hq, hs, topk_ids, topk_w, w2, w2_s, out, id_base):
    lid = nm.to_local(topk_ids, id_base, w2.shape[0])
    y, valid = _y_ref(hq, hs, lid, topk_w, w2, w2_s)
    return (y * valid[:, None]).view(lid.shape[0], lid.shape[1], -1).sum(1)


def _rel(a, b) -> float:
    return nm._rel(a.float(), b)


def _frac(a, b) -> float:
    return float((a.view(torch.uint8) != b.view(torch.uint8)).float().mean()) if a.numel() else 0.0


def route_equal(got, want, e_count: int) -> bool:
    """Equality on what the consumers read: slot_expert and slot_cnt (every
    slot), slot_off of the active slots, the local pairs of pair_list."""
    n_act = int((want[:e_count] >= 0).sum())
    n_loc = int(want[2 * e_count:3 * e_count].sum())
    return (torch.equal(got[:e_count], want[:e_count])
            and torch.equal(got[2 * e_count:3 * e_count], want[2 * e_count:3 * e_count])
            and torch.equal(got[e_count:e_count + n_act], want[e_count:e_count + n_act])
            and torch.equal(got[3 * e_count:3 * e_count + n_loc],
                            want[3 * e_count:3 * e_count + n_loc]))


def _check_route_quant(out, ref):
    aq, a_s, *sections = out
    route = torch.cat(sections)
    aq_r, as_r, route_r, e = ref
    same = route_equal(route, route_r, e)
    fq, scale_eq = _frac(aq, aq_r), bool(torch.equal(a_s, as_r))
    ok = same and scale_eq and fq <= fm.STAGE_TOL["xq"]
    return ok, (f"route equal={same}, x scales bit-equal={scale_eq}, x e4m3 bytes differing "
                f"{fq:.1e} (gate {fm.STAGE_TOL['xq']:.0e}, expected 0)")


def _check_hq(out, ref):
    hq, hs = out
    hq_r, hs_r, rows = ref
    if not bool(rows.any()):
        return True, "no local rows"
    r = _rel(fm.dequant(hq[rows], hs[rows]), fm.dequant(hq_r[rows], hs_r[rows]).double())
    fq = _frac(hq[rows], hq_r[rows])
    ok = r <= 1e-3 and fq <= 1e-2
    return ok, (f"h quant dequant rel {r:.2e} (gate 1e-3), e4m3 bytes differing {fq:.1e} "
                "(f32 vs the spec's f64 h: rare requant flips)")


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
    _entry(FC1, _make_fc1, _ref_fc1, _check_hq, 1, 1, ("moe_fc1_quant", "moe_fc1")),
    _entry(FC2, _make_fc2, _ref_fc2, _check_rows(fm.STAGE_TOL["fc2"], "y"), 1, 2, ("moe_fc2",)),
    _entry(COMBINE, _make_combine, _ref_combine, _check_combine, 1, 3, ("moe_combine",)),
    _entry(FC2_COMBINE, _make_fc2_combine, _ref_fc2_combine, _check_end, 2, 2,
           ("moe_fc2_combine",)),
]
