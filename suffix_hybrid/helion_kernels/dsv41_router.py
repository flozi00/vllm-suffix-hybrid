# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 MoE router in Helion: the gate GEMV, and (negative control) the whole router.

Stock (nightly 39983a5e7), per MoE layer on SM120 (40 backbone layers; the DSpark draft layers
route 128 experts top-3 through another path and are not touched):

  GateLinear  bf16 x bf16 -> f32 logits [M, 384]  (cuteDSL ll_bf16 <= 16 tokens, else cuBLAS)  1 launch
  fused_topk_bias -> dsv4_topk (Triton, 1 program per row, PDL)                                 1 launch

The scoring / bias / top-6 / renormalise / x1.5 kernel is ALREADY one stock launch and writes
exactly what the MoE lane consumes (topk_weights f32 [M, 6], topk_ids i32 [M, 6] GLOBAL ids,
-1 on padding rows; kernels/nvfp4_moe.py takes id - ep_rank*E_local itself), so there is
nothing to convert and the expert-sorting plan stays lane B's (its Helion route root already
ranks the pairs in parallel CTAs inside the launch that quantises x).

What can still be won, in the order the lab should look:
  1. dsv41_gate_gemv        Helion GEMV that replaces the GateLinear GEMM only (stock top-k
                            stays). No split-K, no atomics: f32 accumulation order is fixed by
                            the tile config, so all 8 TP ranks (which recompute the router on
                            identical data and MUST pick identical experts) agree bit for bit.
                            GLM's own router GEMV beat cuBLAS by ~1.4 us/call.
  2. dsv41_gate_topk_resident  GEMV + top-6 in ONE launch (one program per token tile). The fused
                            NEGATIVE CONTROL: it has to stream the 3.9 MB gate weight through
                            one SM per token tile, and the fully fused router kernel already
                            lost on GLM. A cross-CTA fusion (split-K + last-arriver top-k) would
                            need a semaphore, which is banned (grid-level sync deadlocked on
                            silicon). Kept so K4 measures the boundary instead of assuming it.

K1 contract
  x [M, 5120] bf16, w [384, 5120] bf16 (gate.weight), bias f32 [384] (e_score_correction_bias),
  bias_vl f32 [384] (image rows select experts with it), input_ids [M] int (image rows: id in
  [image_lo, image_lo + 5)), pad [M] int8 | bool.
  logits  = x @ w^T                               f32 accumulate
  weights = sqrt(logits > 20 ? logits : log(1 + exp(logits)))
  score   = weights + bias(row)                   NaN -> -1e30, selection only
  sel     = 6 argmax of score, ties -> lowest expert id (dsv4_topk's order)
  w_out_k = weights[sel_k] * routed_scale / sum_k weights[sel_k]
  padding row: ids -1, weights 0
Same math as vllm/model_executor/layers/fused_moe/router/dsv4_topk.py. Numerics vs stock:
the GEMM's f32 summation order and the 6-term weight sum differ (<= 1-2 ulp); the selected
ids are identical unless two scores are within that noise (the same holds between two stock
batch sizes).

Search spaces: gemv bm 16..64 tokens (hl.dot needs >= 16) x bn 16..128 experts x bk 32..256;
resident bt 16..64 x bk 32..256 (E padded to 512 and masked).

Knob: SUFFIX_HELION_DSV41_ROUTER = oxide (default = stock) | auto | helion (the GEMV only).
"""
import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.helion_kernels import dsv41_common as cm

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU host: the module stays importable (lab ABI, tests)
    helion = hl = None
CE = hl.constexpr if hl is not None else int

ENV = "SUFFIX_HELION_DSV41_ROUTER"
K_GEMV = "dsv41_gate_gemv"
K_RES = "dsv41_gate_topk_resident"
MARKER = "[suffix dsv41-router]"
TOPK = 6
EXPERTS = 384
HIDDEN = 5120
IMAGE_LO = 129264    # IMAGE_SENTINEL_BASE_ID (vllm/models/deepseek_v41/common/mm_preprocess.py)
BUCKETS = (1, 2, 4, 6, 12, 24, 48, 96, 192)
MAX_TOKENS = BUCKETS[-1]
SEARCH_SPACE = {K_GEMV: {"bm": (16, 64), "bn": (16, 128), "bk": (32, 256)},
                K_RES: {"bt": (16, 64), "bk": (32, 256)}}


def dsv41_gate_gemv(x: torch.Tensor, w: torch.Tensor):
    """logits f32 [M, N] = x [M, K] bf16 @ w [N, K] bf16 ^ T. One program per (token tile,
    expert tile), full K loop, f32 accumulation: deterministic."""
    M = x.size(0)
    N = hl.specialize(w.size(0))
    K = hl.specialize(x.size(1))
    bm = hl.register_block_size(16, 64)
    bn = hl.register_block_size(16, 128)
    bk = hl.register_block_size(32, 256)
    out = torch.empty([M, N], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([M, N], block_size=[bm, bn]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=bk):
            acc = hl.dot(x[tile_m, tile_k], w[tile_n, tile_k].T, acc=acc)
        out[tile_m, tile_n] = acc
    return out


def dsv41_gate_topk_resident(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor,
                             bias_vl: torch.Tensor, input_ids: torch.Tensor, pad: torch.Tensor,
                             routed_scale: float, image_lo: int, e_pad: CE, has_pad: CE,
                             has_vl: CE):
    """Negative control: GEMV + scoring + top-6 in one launch, one program per token tile.
    (bias_vl / input_ids / pad are ignored when their flag is false but must be tensors.)
    Returns (topk_w f32 [M, 6], topk_ids i32 [M, 6])."""
    M = x.size(0)
    K = hl.specialize(x.size(1))
    E = hl.specialize(w.size(0))
    bt = hl.register_block_size(16, 64)
    bk = hl.register_block_size(32, 256)
    topk_w = torch.empty([M, 6], dtype=torch.float32, device=x.device)
    topk_ids = torch.empty([M, 6], dtype=torch.int32, device=x.device)
    for tile_t in hl.tile(M, block_size=bt):
        for tile_e in hl.tile(E, block_size=e_pad):
            ev = tile_e.index
            lg = hl.zeros([tile_t, tile_e], dtype=torch.float32)
            for tile_k in hl.tile(K, block_size=bk):
                lg = hl.dot(x[tile_t, tile_k], w[tile_e, tile_k].T, acc=lg)
            wgt = torch.sqrt(torch.where(lg > 20.0, lg, torch.log(1.0 + torch.exp(lg))))
            bsel = hl.zeros([tile_t, tile_e], dtype=torch.float32) + bias[tile_e][None, :]
            if has_vl:
                tok = input_ids[tile_t]
                img = (tok >= image_lo) & (tok < image_lo + 5)
                bsel = torch.where(img[:, None], bias_vl[tile_e][None, :], bsel)
            cur = torch.where((ev < E)[None, :], wgt + bsel, float("-inf"))
            cur = torch.where(cur == cur, cur, -1e30)
            cur1 = cur
            wsum = hl.zeros([tile_t], dtype=torch.float32)
            for _ in hl.static_range(6):
                mx = torch.amax(cur1, dim=1)
                eid = torch.amin(torch.where(cur1 == mx[:, None], ev[None, :], E), dim=1)
                hit = ev[None, :] == eid[:, None]
                wsum = wsum + torch.sum(torch.where(hit, wgt, 0.0), dim=1)
                cur1 = torch.where(hit, float("-inf"), cur1)
            fac = routed_scale / torch.where(wsum > 0.0, wsum, 1.0)
            cur2 = cur
            for slot in hl.static_range(6):
                mx = torch.amax(cur2, dim=1)
                eid = torch.amin(torch.where(cur2 == mx[:, None], ev[None, :], E), dim=1)
                hit = ev[None, :] == eid[:, None]
                ow = torch.sum(torch.where(hit, wgt, 0.0), dim=1) * fac
                oid = eid.to(torch.int32)
                if has_pad:
                    isp = pad[tile_t] != 0
                    ow = torch.where(isp, 0.0, ow)
                    oid = torch.where(isp, -1, oid)
                topk_w[tile_t, slot] = ow
                topk_ids[tile_t, slot] = oid
                cur2 = torch.where(hit, float("-inf"), cur2)
    return topk_w, topk_ids


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------
def _e_pad(e: int) -> int:
    return 1 << (e - 1).bit_length()


def check_operands(x, w):
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"{MARKER} need bf16 x / bf16 gate weight, got {x.dtype} / {w.dtype}")
    if x.dim() != 2 or w.dim() != 2 or x.shape[1] != w.shape[1] or not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError(f"{MARKER} x {tuple(x.shape)} / w {tuple(w.shape)} not a contiguous [M,K] x [N,K]")
    if w.shape[0] > 1024:
        raise ValueError(f"{MARKER} {w.shape[0]} experts: the resident kernel pads E to a power of two <= 1024")


def resident_args(x, w, bias, bias_vl, input_ids, is_padding, routed_scale, image_lo=0):
    check_operands(x, w)
    has_vl = bias_vl is not None and image_lo > 0
    has_pad = is_padding is not None
    return (x, w, bias, bias_vl if has_vl else bias, input_ids if input_ids is not None else bias,
            is_padding if has_pad else bias, float(routed_scale), int(image_lo), _e_pad(w.shape[0]),
            has_pad, has_vl)


def case(tokens: int) -> dict:
    return {"numtokens": tokens, "experts": EXPERTS, "hidden": HIDDEN}


def serving_case(tokens: int) -> dict:
    return case(hk.bucket(tokens, BUCKETS))


_WARM: set = set()


def gate_logits(x, w):
    """The GateLinear GEMM replacement: f32 [M, N] logits (hook: kernels/dsv41_fuse_hook.py)."""
    check_operands(x, w)
    c = serving_case(x.shape[0])
    key = (K_GEMV, c["numtokens"], tuple(w.shape))
    if key not in _WARM:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{MARKER} {key} first used while capturing a CUDA graph: warm it eagerly")
        _WARM.add(key)
    return hk.compiled(dsv41_gate_gemv, K_GEMV, c)(x, w)


def missing() -> list:
    return [b for b in BUCKETS if hk.config(K_GEMV, case(b)) is None]


# ---------------------------------------------------------------------------
# exact references (float64) + the tie-aware judge
# ---------------------------------------------------------------------------
def logits_reference(x, w):
    return (x.double() @ w.double().T)


def scores_ref(x, w, bias, bias_vl, input_ids, image_lo):
    lg = logits_reference(x, w)
    wgt = torch.sqrt(torch.where(lg > 20.0, lg, torch.log(1.0 + torch.exp(lg))))
    b = bias.double().expand_as(wgt).clone()
    if bias_vl is not None and image_lo > 0 and input_ids is not None:
        img = (input_ids >= image_lo) & (input_ids < image_lo + 5)
        b[img] = bias_vl.double()
    return wgt, wgt + b


def reference(x, w, bias, bias_vl, input_ids, pad, routed_scale, image_lo, e_pad, has_pad, has_vl):
    wgt, score = scores_ref(x, w, bias, bias_vl if has_vl else None, input_ids,
                            image_lo if has_vl else 0)
    # stable top-6: highest score, ties -> lowest id (a stable sort of -score)
    ids = torch.sort(-score, dim=1, stable=True).indices[:, :TOPK]
    sel = wgt.gather(1, ids)
    ow = sel * (routed_scale / sel.sum(1, keepdim=True).clamp_min(1e-300))
    ids32 = ids.to(torch.int32)
    if has_pad:
        p = (pad != 0)[:, None]
        ow = torch.where(p, torch.zeros_like(ow), ow)
        ids32 = torch.where(p, torch.full_like(ids32, -1), ids32)
    return ow.float(), ids32, score


def judge(got, ref):
    """Tie-aware: rows whose 6th/7th f64 scores are within ~1e-4 are not judged on membership
    (the f32 GEMM order may flip them, as it may between stock batch sizes); all other rows
    must pick the exact set in the same order, and every chosen weight must match to rel 2e-5."""
    (gw, gi), (rw, ri, score) = got, ref
    srt = torch.sort(score, dim=1, descending=True).values
    gap = (srt[:, TOPK - 1] - srt[:, TOPK]).abs()
    gap_in = (srt[:, :TOPK - 1] - srt[:, 1:TOPK]).abs().min(1).values
    clear = (gap > 1e-4) & (gap_in > 1e-4)
    pad = (ri < 0).all(1)
    same = (gi == ri).all(1)
    bad_ids = int((~same & clear).sum())
    sel = clear & same
    rel = float(((gw - rw).abs() / rw.abs().clamp_min(1e-9))[sel].max()) if bool(sel.any()) else 0.0
    ok = bad_ids == 0 and rel < 2e-5 and bool(gw.isfinite().all())
    if bool(pad.any()):
        ok = ok and bool((gw[pad] == 0).all()) and bool((gi[pad] == -1).all())
    return ok, f"{int(clear.sum())}/{len(clear)} rows judged, {bad_ids} id-mismatch, weight rel {rel:.1e}"


def judge_logits(got, x, w):
    ref = logits_reference(x, w)
    rel = float((got.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
    return rel < 1e-5 and bool(got.isfinite().all()), f"logits rel {rel:.1e}"


# ---------------------------------------------------------------------------
# problems / lab ABI
# ---------------------------------------------------------------------------
def make_problem(tokens: int, seed: int = 0, device: str = "cpu", experts: int = EXPERTS,
                 hidden: int = HIDDEN, with_pad: bool = True, scale: float = 0.03):
    """Gate-like data: x ~ N(0,1), w ~ N(0, scale) so logits are O(2) (softplus regime), bias ~
    N(0, 0.2); ~1/3 image-sentinel rows; the last row is padding."""
    g = torch.Generator().manual_seed(77 + seed)
    m = tokens
    x = torch.randn(m, hidden, generator=g).to(torch.bfloat16).to(device)
    w = (torch.randn(experts, hidden, generator=g) * scale).to(torch.bfloat16).to(device)
    bias = (torch.randn(experts, generator=g) * 0.2).to(device)
    bias_vl = (torch.randn(experts, generator=g) * 0.2).to(device)
    ids = torch.randint(0, 129000, (m,), generator=g)
    ids[::3] = IMAGE_LO + torch.arange(len(ids[::3])) % 5
    pad = torch.zeros(m, dtype=torch.int8)
    if with_pad and m > 2:
        pad[-1] = 1
    return x, w, bias, bias_vl, ids.to(device), pad.to(device) if with_pad else None


def _is_v41(model: dict) -> bool:
    return str(model.get("architecture", "")).startswith(("DeepseekV41", "DeepseekV4"))


def cases(model: dict) -> list:
    return [case(b) for b in BUCKETS] if _is_v41(model) else []


def default_config(case_: dict, kernel: str = K_GEMV) -> dict:
    """A starting point (NOT tuned) inside the bounded search space."""
    t = case_["numtokens"]
    if kernel == K_GEMV:
        return {"block_sizes": [16 if t <= 16 else 32, 32, 128], "num_warps": 4, "num_stages": 3}
    return {"block_sizes": [16, 64], "num_warps": 4}


def _make_gemv(c, seed):
    x, w, *_ = make_problem(c["numtokens"], seed, cm.DEVICE)
    return (x, w)


def _make_resident(c, seed):
    x, w, bias, bias_vl, ids, pad = make_problem(c["numtokens"], seed, cm.DEVICE)
    return resident_args(x, w, bias, bias_vl, ids, pad, 1.5, IMAGE_LO)


def _cpu(t):
    return tuple(a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in t)


_LAST: dict = {}


def _gemv_baseline(x, w):
    """Stock GateLinear: cuteDSL ll_bf16 GEMV up to 16 tokens, cuBLAS bf16->f32 above."""
    if x.shape[0] <= 16:
        try:
            from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import ll_bf16_gemm
            return ll_bf16_gemm(x, w)
        except ImportError:
            pass
    return torch.mm(x, w.T, out_dtype=torch.float32)


def _resident_baseline(x, w, bias, bias_vl, input_ids, pad, routed_scale, image_lo, e_pad, has_pad,
                       has_vl):
    """Stock: the gate GEMM, then vLLM's dsv4_topk Triton kernel (2 launches)."""
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk
    return dsv4_topk(_gemv_baseline(x, w), bias, torch.int32, routed_scale,
                     is_padding=(pad != 0) if has_pad else None,
                     input_ids=input_ids if has_vl else None,
                     bias_vl=bias_vl if has_vl else None, image_sentinel_lo=image_lo if has_vl else 0)


def _lab_make(kind):
    make = {"gemv": _make_gemv, "res": _make_resident}[kind]

    def wrapped(c, seed):
        args = make(c, seed)
        _LAST[kind] = args
        return args
    return wrapped


KERNELS = [
    {"name": K_GEMV, "body": dsv41_gate_gemv, "cases": cases, "make": _lab_make("gemv"),
     "reference": lambda x, w: logits_reference(x, w).float(),
     "baseline": _gemv_baseline,
     "baseline_name": "GateLinear: cuteDSL ll_bf16 GEMV (<= 16 tokens) / cuBLAS bf16->f32 (above)",
     "check": lambda got, ref: judge_logits(got.cpu(), *_cpu(_LAST["gemv"])),
     "replaces": [r"ll_bf16", r"nvjet.*(?:f32|fp32)|gemv.*bf16"]},
    {"name": K_RES, "body": dsv41_gate_topk_resident, "cases": cases, "make": _lab_make("res"),
     "reference": lambda *a: reference(*a),
     "baseline": _resident_baseline,
     "baseline_name": "gate GEMM + vLLM dsv4_topk (Triton, PDL): 2 launches",
     "check": lambda got, ref: judge(_cpu(got), _cpu(ref)),
     "replaces": [r"_dsv4_topk_kernel", r"ll_bf16"]},
]


# ---------------------------------------------------------------------------
# boot gates <kernel>_capture_replay
# ---------------------------------------------------------------------------
def _built(name, body, case_, cfg=None):
    import helion as _h
    cfg = cfg or hk.config(name, case_) or default_config(case_, name)
    return _h.kernel(body, config=_h.Config(**cfg), static_shapes=False)


def dsv41_gate_gemv_capture_replay(tokens: int = 6, cfg=None, runner_factory=None) -> bool:
    kern = _built(K_GEMV, dsv41_gate_gemv, serving_case(tokens), cfg)

    def make(i):
        x, w, *_ = make_problem(tokens, i, cm.DEVICE)
        return [x, w]
    return cm.capture_replay_gate(K_GEMV, lambda x, w: kern(x, w), make, runner_factory=runner_factory)


def dsv41_gate_topk_resident_capture_replay(tokens: int = 6, cfg=None, runner_factory=None) -> bool:
    kern = _built(K_RES, dsv41_gate_topk_resident, serving_case(tokens), cfg)
    sample = _make_resident(case(tokens), 0)

    def make(i):
        x, w, bias, bias_vl, ids, pad = make_problem(tokens, i, cm.DEVICE)
        return [x, w, bias, bias_vl, ids, pad]
    return cm.capture_replay_gate(
        K_RES, lambda x, w, bias, bias_vl, ids, pad: kern(x, w, bias, bias_vl, ids, pad, *sample[6:]), make,
        runner_factory=runner_factory)


BOOT_GATES = {f"{K_GEMV}_capture_replay": dsv41_gate_gemv_capture_replay,
              f"{K_RES}_capture_replay": dsv41_gate_topk_resident_capture_replay}


def bench_boundary(tokens=BUCKETS, iters: int = 100, cfg_file=None) -> int:
    """K4 for the fusion boundary the lab cannot see from one entry: the whole router, per M bucket,
    cold L2, CUDA graph:  stock (GateLinear GEMM + dsv4_topk)  vs  unfused-Helion (dsv41_gate_gemv +
    stock dsv4_topk, 2 launches)  vs  fused-resident (dsv41_gate_topk_resident, 1 launch).
    Needs SM120 + the vLLM image. --config FILE = {"<kernel>:<tokens>": {helion.Config kwargs}}."""
    import json

    import helion as _h

    from suffix_hybrid.kernels import dsv41_bench as bench
    table = json.load(open(cfg_file)) if cfg_file else {}
    status = 0
    print("== dsv41 router: stock GEMM+dsv4_topk vs Helion GEMV+dsv4_topk vs resident GEMV+top-k (us, cold L2)")
    for t in tokens:
        c = case(t)
        x, w, bias, bvl, ids, pad = make_problem(t, 0, cm.DEVICE)
        args = resident_args(x, w, bias, bvl, ids, pad, 1.5, IMAGE_LO)
        gemv = _h.kernel(dsv41_gate_gemv, static_shapes=False, config=_h.Config(
            **(table.get(f"{K_GEMV}:{t}") or hk.config(K_GEMV, c) or default_config(c, K_GEMV))))
        res = _h.kernel(dsv41_gate_topk_resident, static_shapes=False, config=_h.Config(
            **(table.get(f"{K_RES}:{t}") or hk.config(K_RES, c) or default_config(c, K_RES))))
        from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

        def unfused():
            return dsv4_topk(gemv(x, w), bias, torch.int32, 1.5, is_padding=(pad != 0),
                             input_ids=ids, bias_vl=bvl, image_sentinel_lo=IMAGE_LO)
        ref = reference(*args)
        ok1, m1 = judge(unfused(), ref)
        ok2, m2 = judge(res(*args), ref)
        t_stock = bench.graph_us(lambda: _resident_baseline(*args), iters)
        t_un, t_res = bench.graph_us(unfused, iters), bench.graph_us(lambda: res(*args), iters)
        print(f"  T={t:<4d} stock {t_stock:7.1f} us | unfused {t_un:7.1f} us x{t_stock / t_un:4.2f} "
              f"{'ok' if ok1 else 'FAIL ' + m1} | resident {t_res:7.1f} us x{t_stock / t_res:4.2f} "
              f"{'ok' if ok2 else 'FAIL ' + m2}", flush=True)
        status |= 0 if (ok1 and ok2) else 1
    return status


def main(argv=None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog=__name__)
    ap.add_argument("cmd", choices=["oracle", "bench", "capture_replay"])
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--config")
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--buckets", type=int, nargs="*")
    ap.add_argument("--iters", type=int, default=100)
    ns, _ = ap.parse_known_args(argv)
    if ns.cmd == "capture_replay":
        return 0 if all(g(tokens=ns.tokens) for g in BOOT_GATES.values()) else 1
    if ns.cmd == "bench" and not ns.autotune:
        return bench_boundary(tuple(ns.buckets or BUCKETS), ns.iters, ns.config)
    from suffix_hybrid.kernels import dsv41_bench
    return dsv41_bench.main(sys.modules[__name__], argv)


if __name__ == "__main__":
    raise SystemExit(main())
