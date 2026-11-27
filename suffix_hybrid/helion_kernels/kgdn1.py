# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 on Helion: the fused Gated-DeltaNet non-spec decode step
(port of kernels-oxide/kgdn1, PTX entry kgdn1_decode_h16_hv48_k128).

One step per decode token t and value head hv (key head h = hv // G, G = HV/H):

    q, k   = L2norm(q_h) * K^-0.5, L2norm(k_h)          (bf16 -> f32, eps 1e-6)
    decay  = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b)
    s      = decay * S[slot, hv]                        S: f32 [V, K], in place
    dv     = beta * (v - s @ k);  S[slot, hv] = s + dv k^T
    o      = s @ q + dv (k . q)                         (= S_new @ q, f32)
    out    = bf16(o * rsqrt(mean(o^2) + eps) * norm_w * gate(z)),  gate = silu | sigmoid

slot = state_idx[t]; slot <= 0 is vLLM's NULL block: zero row, no state IO.
Semantics, layouts and the oracle are the lane's (suffix_hybrid/kernels/
qwen_gdn.py); the numpy twin of the oxide kernel is kernels/qwen_gdn_oxide.py.

Design: one program per (t, h, g) holding the whole [V, K] state tile —
ONE read + ONE write of the state, every reduction (pk, pq, the RMS norm over
V) in registers, no scratch. The body is shape-generic (any H, G, K, V; the
oxide entry is hard-coded h16/hv48/k128), so every Qwen GDN rank layout runs
it. Tunables the lab searches (Helion 1.5 config spec of this body): num_warps
/ num_stages (how the 64 KiB tile spreads over registers), pid_type (flat |
persistent), indexing per q/k/v/z/out access (pointer | block_ptr |
tensor_descriptor; the slot-gathered state is always pointer) and eviction
policy per load.
ponytail: V is never split across programs because the RMS norm couples all
V rows; at T=1 that is HV programs (48 on the 27B, like oxide; FLA runs 4x
more 1-warp programs). Upgrade path if T=1 loses in the lab: a split-V phase
writing f32 o + hl.barrier() + a norm phase (persistent kernel).

Serving: ``decode_op`` (called by the lane through qwen_gdn.decode_op only
when SUFFIX_HELION_KGDN1 != oxide) returns a launcher with the oxide op's
exact call signature, routed per numtokens bucket.
"""
from __future__ import annotations

import re

import torch

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.kernels import qwen_gdn

hk.ensure_vendor()
try:
    import helion
    import helion.language as hl
except ImportError:  # CPU / no-Helion host: module stays importable (lab ABI, tests)
    helion = hl = None

ENV = "SUFFIX_HELION_KGDN1"
KERNEL = "kgdn1_decode"
BUCKETS = (1, 8, 32, 128)  # numtokens the lab tunes; bucket() maps serving T onto them
OXIDE_ENTRY = "kgdn1_decode_h16_hv48_k128"
OXIDE_SHAPE = (16, 48, 128)  # (H, HV, K) of the shipped oxide cubin
INT32_MAX = 2 ** 31 - 1

# GDN dims are not in the console's model_dims(): per architecture, keyed by
# hidden size, (linear key heads, linear value heads, head_k_dim, head_v_dim)
# of the FULL model, mirroring the checkpoints' config.json. An unknown hidden
# size yields no cases (never guessed dims).
GDN_DIMS = {
    # qwen3.8-27b(-fable-distill)(-nvfp4): Qwen3.5-27B layout
    "Qwen3_5ForConditionalGeneration": {5120: (16, 48, 128, 128)},
    # Qwen3.5-35B-A3B
    "Qwen3_5MoeForConditionalGeneration": {2048: (16, 32, 128, 128)},
    # qwen3.8-flash-next-*: Qwen3-Next-80B-A3B layout
    "Qwen3NextForCausalLM": {2048: (16, 32, 128, 128)},
}


# ---------------------------------------------------------------------------
# Helion body (module-level, undecorated: the lab and compiled() wrap it)
# ---------------------------------------------------------------------------
def kgdn1_decode(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, out,
                 scale: float, eps: float, act: hl.constexpr):
    """q, k [T, H, K] bf16; v, z [T, H, G, V] bf16; b, a [T, H, G] bf16;
    a_log, dt_bias [H, G] f32; norm_w [V] f32; state [S, H, G, V, K] f32
    (updated in place); state_idx [T] int32; out [T, H, G, V] bf16 (written).
    act 0 = silu gate, 1 = sigmoid gate. Returns (out, state)."""
    T, H, K = q.shape
    G, V = v.shape[2], v.shape[3]
    K = hl.specialize(K)
    V = hl.specialize(V)
    for t, h, g in hl.grid([T, H, G]):
        slot = state_idx[t]
        # ponytail: no slot < S bound (oxide traps there; Helion has no trap,
        # and vLLM's FLA kernel has no bound either). A bound would make S a
        # Triton int arg whose %16 class re-JITs on the serving pool size.
        if slot > 0:
            qf = q[t, h, :].to(torch.float32)
            kf = k[t, h, :].to(torch.float32)
            qf = qf * (scale / torch.sqrt(torch.sum(qf * qf, dim=-1) + 1e-6))
            kf = kf / torch.sqrt(torch.sum(kf * kf, dim=-1) + 1e-6)
            kq = torch.sum(kf * qf, dim=-1)
            x = a[t, h, g].to(torch.float32) + dt_bias[h, g]
            sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
            decay = torch.exp(-torch.exp(a_log[h, g]) * sp)
            beta = torch.sigmoid(b[t, h, g].to(torch.float32))
            s = state[slot, h, g, :, :] * decay
            pk = torch.sum(s * kf[None, :], dim=-1)
            pq = torch.sum(s * qf[None, :], dim=-1)
            dv = beta * (v[t, h, g, :].to(torch.float32) - pk)
            state[slot, h, g, :, :] = s + dv[:, None] * kf[None, :]
            o = pq + dv * kq
            rstd = torch.rsqrt(torch.mean(o * o, dim=-1) + eps)
            zf = z[t, h, g, :].to(torch.float32)
            if act == 0:
                gate = zf * torch.sigmoid(zf)
            else:
                gate = torch.sigmoid(zf)
            out[t, h, g, :] = (o * rstd * norm_w[:] * gate).to(out.dtype)
        else:
            out[t, h, g, :] = hl.zeros([V], dtype=out.dtype)
    return out, state


def body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads):
    """The oxide op's tensors as the body's head-split views (zero copies):
    mixed_qkv [T, 2HK+HV*V] -> q, k, v; ba [T, 2HV] -> b, a; HV -> (H, G)."""
    T = mixed_qkv.shape[0]
    S, HV, V, K = state.shape
    H = num_k_heads
    G = HV // H
    q = mixed_qkv[:, :H * K].unflatten(1, (H, K))
    k = mixed_qkv[:, H * K:2 * H * K].unflatten(1, (H, K))
    v = mixed_qkv[:, 2 * H * K:2 * H * K + HV * V].unflatten(1, (H, G, V))
    return (q, k, v, ba[:, :HV].unflatten(1, (H, G)), ba[:, HV:2 * HV].unflatten(1, (H, G)),
            z.unflatten(1, (H, G)), a_log.view(H, G), dt_bias.view(H, G), norm_w,
            state.unflatten(1, (H, G)), state_idx, out.view(T, H, G, V))


def case(numtokens, h, hv, k, v) -> dict:
    return {"numtokens": numtokens, "h": h, "hv": hv, "k": k, "v": v}


# ---------------------------------------------------------------------------
# serving (the lane: suffix_hybrid/kernels/qwen_gdn.py)
# ---------------------------------------------------------------------------
def gdn_decode_fused_helion(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx,
                            out, num_k_heads, scale, norm_eps, act):
    """One K-GDN1 step on Helion (the bucket's shipped config; KeyError if none)."""
    T = mixed_qkv.shape[0]
    if T == 0:
        return None
    S, HV, V, K = state.shape
    # Helion indexes in int32 here (dynamic shapes size the choice by numel,
    # not by vLLM's padded page): refuse a pool whose slot offsets overflow.
    if (S - 1) * state.stride(0) + HV * V * K > INT32_MAX:
        raise RuntimeError(f"{hk.MARKER} {KERNEL}: state pool spans > 2^31 elements "
                           f"(S={S}, page {state.stride(0)}); run {ENV}=oxide")
    kern = hk.compiled(kgdn1_decode, KERNEL, case(hk.bucket(T, BUCKETS), num_k_heads, HV, K, V))
    kern(*body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out,
                    num_k_heads), float(scale), float(norm_eps), int(act))
    return None


def decode_op(native, h, hv, k, v):
    """(op, helion_buckets) for one layer shape under ENV (helion | auto).

    op has the oxide op's signature (``native.gdn_decode_fused_cuda``, whose
    stream_ptr is torch's current stream — the stream Helion launches on).
    helion: every bucket needs a shipped config, else RuntimeError now (load).
    auto: Helion where a config is shipped (= it beat oxide in the lab), oxide
    elsewhere."""
    mode = hk.backend(ENV)
    use = {}
    for m in BUCKETS:
        c = case(m, h, hv, k, v)
        use[m] = hk.config(KERNEL, c) is not None
        if mode == "helion" and not use[m]:
            raise RuntimeError(
                f"{ENV}=helion but no tuned {KERNEL} config for {hk.case_key(c)} on "
                f"{hk.platform()} (tune it in the autotune lab, ship configs/helion/{KERNEL}/"
                f"<platform>.json, or use {ENV}=auto|oxide)")
    oxide = native.gdn_decode_fused_cuda
    buckets = tuple(m for m in BUCKETS if use[m])
    if not buckets:
        return oxide, ()

    def op(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads,
           scale, norm_eps, act, stream_ptr):
        if use[hk.bucket(mixed_qkv.shape[0], BUCKETS)]:
            return gdn_decode_fused_helion(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state,
                                           state_idx, out, num_k_heads, scale, norm_eps, act)
        return oxide(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out,
                     num_k_heads, scale, norm_eps, act, stream_ptr)

    return op, buckets


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def cases(model) -> list:
    """Per-TP-rank GDN head layout x numtokens buckets; [] for non-GDN models."""
    dims = GDN_DIMS.get(model.get("architecture"), {}).get(model.get("hidden"))
    tp = model.get("tp") or 1
    if dims is None or dims[0] % tp or dims[1] % tp or dims[2] != dims[3]:
        return []  # dims[2] != dims[3]: the lane's make_inputs is K == V
    return [case(m, dims[0] // tp, dims[1] // tp, dims[2], dims[3]) for m in BUCKETS]


def problem(c, seed, device="cuda"):
    """The oxide op's args at case `c`: the lane's make_inputs (bf16 row-strided
    qkvz views, fp32 state in vLLM's padded pages), distinct live slots in
    random order, act = silu (every Qwen GDN checkpoint)."""
    T, H, HV, K = c["numtokens"], c["h"], c["hv"], c["k"]
    slots = T + 1
    gen = torch.Generator().manual_seed(7919 + seed)
    idx = (torch.randperm(slots - 1, generator=gen)[:T] + 1).tolist()
    inp = qwen_gdn.make_inputs(T, H, HV, K, slots, device=device, seed=seed, idx=idx,
                               page_pad=4096)
    out = torch.zeros(T, HV, K, dtype=torch.bfloat16, device=device)
    return (inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], out, H, K ** -0.5, 1e-6, 0)


def make(c, seed, device="cuda"):
    *tensors, num_k_heads, scale, eps, act = problem(c, seed, device)
    return (*body_args(*tensors, num_k_heads), scale, eps, act)


def make_baseline(c, seed):
    args = problem(c, seed)
    H, HV, K = args[9], args[6].shape[1], args[6].shape[3]
    if (H, HV, K) == OXIDE_SHAPE:
        qwen_gdn.load_prebuilt(H, HV, K)  # driver-load the cubin outside graph capture
    return args


def baseline(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads,
             scale, eps, act):
    """The oxide op where its cubin's shape fits, else vLLM's stock chain."""
    if (num_k_heads, state.shape[1], state.shape[3]) == OXIDE_SHAPE:
        qwen_gdn._native().gdn_decode_fused_cuda(
            mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads,
            scale, eps, act, torch.cuda.current_stream().cuda_stream)
    else:
        qwen_gdn.stock_decode_fused(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state,
                                    state_idx, out, num_k_heads, scale, eps, act)
    return out, state


def reference(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, out, scale, eps,
              act):
    """Exact eager answer (vLLM semantics = the oxide twin's, f32 o into the
    norm), vectorized over tokens and heads. Does not mutate its args."""
    idx = state_idx.long()
    live = idx > 0
    rows = idx[live]
    qf, kf = q[live].float(), k[live].float()
    qf = qf * (scale / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6))
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    x = a[live].float() + dt_bias
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    decay = torch.exp(-torch.exp(a_log) * sp)
    beta = torch.sigmoid(b[live].float())
    kk, qq = kf[:, :, None, None, :], qf[:, :, None, None, :]   # [n, H, 1, 1, K]
    s = state[rows].float() * decay[..., None, None]             # [n, H, G, V, K]
    dv = beta[..., None] * (v[live].float() - (s * kk).sum(-1))
    s = s + dv[..., None] * kk
    new_state = state.clone()
    new_state[rows] = s
    o = (s * qq).sum(-1)                                          # [n, H, G, V]
    zf = z[live].float()
    gate = zf * torch.sigmoid(zf) if act == 0 else torch.sigmoid(zf)
    y = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + eps) * norm_w * gate
    ref_out = torch.zeros(out.shape, dtype=out.dtype, device=out.device)
    ref_out[live] = y.to(out.dtype)
    return ref_out, new_state


def check(got, ref):
    """The lane's oracle gate: out within ~2 bf16 ulps, state f32-tight."""
    (out, state), (ref_out, ref_state) = got, ref
    o, r = out.float(), ref_out.float()
    d_out = (o - r).abs()
    d_state = (state - ref_state).abs()
    bad = int((d_out > qwen_gdn.OUT_ATOL + qwen_gdn.OUT_RTOL * r.abs()).sum())
    bad_s = int((d_state > qwen_gdn.STATE_ATOL + qwen_gdn.STATE_RTOL * ref_state.abs()).sum())
    ok = bad == 0 and bad_s == 0 and bool(o.isfinite().all())
    return ok, (f"out max|d|={float(d_out.max()):.3e} ({bad} bad), "
                f"state max|d|={float(d_state.max()):.3e} ({bad_s} bad)")


KERNELS = [
    {"name": KERNEL, "body": kgdn1_decode, "cases": cases, "make": make,
     "make_baseline": make_baseline, "reference": reference, "baseline": baseline,
     "baseline_name": (f"oxide {OXIDE_ENTRY} (h16/hv48/k128 cases); vLLM FLA packed decode "
                       "+ layer_norm_fwd chain (shapes the oxide cubin cannot run)"),
     "check": check,
     "replaces": [re.escape(OXIDE_ENTRY), r"fused_recurrent_gated_delta_rule_packed_decode_kernel",
                  r"^layer_norm_fwd_kernel"]},
]
