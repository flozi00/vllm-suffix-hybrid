# SPDX-License-Identifier: Apache-2.0
"""K-GDN1 on Helion: the fused Gated-DeltaNet decode step (port of
kernels-oxide/kgdn1, PTX entry kgdn1_decode_h16_hv48_k128) and the MTP
spec-verify step (vLLM's fused_gdn_decode_post_conv_mtp semantics).

One step per token t and value head hv (key head h = hv // G, G = HV/H):

    q, k   = L2norm(q_h) * K^-0.5, L2norm(k_h)          (bf16 -> f32, eps 1e-6)
    decay  = exp(-exp(A_log) * softplus(a + dt_bias)),  beta = sigmoid(b)
    s      = decay * S                                  S: f32 [V, K]
    dv     = beta * (v - s @ k);  S' = s + dv k^T
    o      = s @ q + dv (k . q)                         (= S' @ q, f32)
    out    = bf16(o * rsqrt(mean(o^2) + eps) * norm_w * gate(z)),  gate = silu | sigmoid

kgdn1_decode (non-spec decode, one token per sequence): S = state[slot],
slot = state_idx[t], S' written back in place; slot <= 0 is vLLM's NULL
block: zero row, no state IO.

kgdn1_verify (MTP verify, up to W = k+1 tokens per sequence n, tokens
cu[n] .. cu[n+1]-1): S starts at state[state_idx[n, acc[n]-1]] (the state
after the last accepted token of the previous step) and runs the recurrence
token by token; the state after token t goes to state[state_idx[n, t]] (if
> 0), so whatever the sampler accepts, its state already sits in a slot and
rejected drafts need no rollback (next step reads slot acc-1 again). Source
slot <= 0 or acc outside 1..W: the sequence's rows are zeroed, no state IO
(vLLM's CUDA MTP kernel; padded cudagraph rows are NULL with 0 tokens).

Semantics, layouts and the oracle are the lane's (suffix_hybrid/kernels/
qwen_gdn.py); the numpy twin of the oxide kernel is kernels/qwen_gdn_oxide.py.

Design: one program per (sequence/token, h, g) holding the whole [V, K]
state tile — ONE read of the state (+ one write per token), every reduction
(pk, pq, the RMS norm over V) in registers, no scratch. The bodies are
shape-generic (any H, G, K, V, W — every TP rank layout); q/k/v/z/b/a are
strided views, so the flat (Qwen3.5) and the de-interleaved (Qwen3-Next) z
both run without a copy. Tunables the lab searches (Helion 1.5 config spec):
num_warps / num_stages, pid_type, indexing per access, eviction policies.
At T=1 that is HV programs (48 on the 27B TP1, 24 at TP2): small batches
lose to vLLM's CUDA MTP kernel, which splits V across blocks.

V-tiled bodies (kgdn1_decode_vt / kgdn1_verify_vt): the parallel grid is
(token | sequence, h, g, V tile), V tile = the tunable ``v_block`` (16 / 32 /
64). The delta-rule update is row-independent (pk, pq: per-row dots over K;
q/k norms and k.q recomputed per program), so each program reads and writes
only its V rows of the state (verify: same slot semantics, token loop per
program). The RMS norm couples all V rows, so the bodies only write
unnormalized f32 o to a per-call ``torch.empty`` scratch (graph-private under
capture, no module-level buffer) and return (o, state); nothing runs after
their device loop. The gated norm is its own Helion kernel
(kgdn1_vt_norm / kgdn1_vt_norm_seq at NORM_CONFIG, one [V] row per program,
out = bf16(o * rsqrt(mean(o^2) + eps) * norm_w * gate)), launched by plain
Python right after the body (``vt_post``). Two launches, no nesting (Helion
refuses a kernel call inside a body, and a host-side op that compiles a
Helion kernel fails under the real compiler's host trace). The tuner's
optional KERNELS ``post`` field makes it check and time body + post while it
autotunes only the body.

Serving: ``decode_op`` / ``verify_op`` (called by the lane only when
SUFFIX_HELION_KGDN1 != oxide) route per numtokens / sequence bucket.
Body per bucket (``pick``): the full-tile body when its config is shipped,
else the V-tiled body when its config is shipped, else the baseline. Config
rows carry no timing, so where both bodies win the config pull keeps only
the faster body's row (drop the full-tile row to serve the V-tiled one);
with no *_vt row shipped serving is exactly the full-tile-only behaviour.
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
    from helion.autotuner import PowerOfTwoFragment
except ImportError:  # CPU / no-Helion host: module stays importable (lab ABI, tests)
    helion = hl = PowerOfTwoFragment = None

ENV = "SUFFIX_HELION_KGDN1"
KERNEL = "kgdn1_decode"
VERIFY_KERNEL = "kgdn1_verify"
KERNEL_VT = "kgdn1_decode_vt"          # V-tiled bodies (+ the gated-norm launch, vt_post)
VERIFY_KERNEL_VT = "kgdn1_verify_vt"
# The norm launch's fixed config (never tuned: one [V] row per program).
# ponytail: fixed num_warps; give it a KERNELS entry if the lab shows it matters.
NORM_CONFIG = {"num_warps": 4}
BUCKETS = (1, 8, 32, 128)  # decode: numtokens; verify: sequences (case numtokens = n * w)
TPS = (1, 2, 4, 8)         # cases() without a model tp: every TP whose heads divide
OXIDE_ENTRY = "kgdn1_decode_h16_hv48_k128"
OXIDE_SHAPE = (16, 48, 128)  # (H, HV, K) of the shipped oxide cubin (K == V, flat layout)
INT32_MAX = 2 ** 31 - 1

# Fallback for callers whose model dict lacks the linear_* fields: per
# architecture, keyed by hidden size, (linear key heads, linear value heads,
# head_k_dim, head_v_dim) of the FULL model (checkpoints' config.json). An
# unknown hidden size yields no cases (never guessed dims).
GDN_DIMS = {
    # qwen3.8-27b(-fable-distill)(-nvfp4): Qwen3.5-27B layout
    "Qwen3_5ForConditionalGeneration": {5120: (16, 48, 128, 128)},
    # Qwen3.5-35B-A3B
    "Qwen3_5MoeForConditionalGeneration": {2048: (16, 32, 128, 128)},
    # qwen3.8-flash-next-*: Qwen3-Next-80B-A3B layout
    "Qwen3NextForCausalLM": {2048: (16, 32, 128, 128)},
}
LINEAR_FIELDS = ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                 "linear_value_head_dim")


# ---------------------------------------------------------------------------
# Helion bodies (module-level, undecorated: the lab and compiled() wrap them)
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


def kgdn1_verify(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, cu_lo, cu_hi,
                 acc, out, scale: float, eps: float, act: hl.constexpr):
    """kgdn1_decode's tensors, but state_idx [N, W] int32 (per-position slots
    of each spec sequence), cu_lo / cu_hi [N] int32 (= cu_seqlens[:-1] /
    [1:]: the sequence's token rows), acc [N] int32 (num_accepted_tokens).
    Rows outside every sequence are not written. Returns (out, state)."""
    T, H, K = q.shape
    G, V = v.shape[2], v.shape[3]
    N, W = state_idx.shape
    K = hl.specialize(K)
    V = hl.specialize(V)
    W = hl.specialize(W)
    for n, h, g in hl.grid([N, H, G]):
        bos = cu_lo[n]
        tn = cu_hi[n] - bos
        na = acc[n]
        src = state_idx[n, torch.clamp(na - 1, 0, W - 1)]
        if (src > 0) & (na >= 1) & (na <= W):
            s = state[src, h, g, :, :]
            neg_a = -torch.exp(a_log[h, g])
            for i in hl.static_range(W):
                if i < tn:
                    row = bos + i
                    qf = q[row, h, :].to(torch.float32)
                    kf = k[row, h, :].to(torch.float32)
                    qf = qf * (scale / torch.sqrt(torch.sum(qf * qf, dim=-1) + 1e-6))
                    kf = kf / torch.sqrt(torch.sum(kf * kf, dim=-1) + 1e-6)
                    kq = torch.sum(kf * qf, dim=-1)
                    x = a[row, h, g].to(torch.float32) + dt_bias[h, g]
                    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
                    beta = torch.sigmoid(b[row, h, g].to(torch.float32))
                    s = s * torch.exp(neg_a * sp)
                    pk = torch.sum(s * kf[None, :], dim=-1)
                    pq = torch.sum(s * qf[None, :], dim=-1)
                    dv = beta * (v[row, h, g, :].to(torch.float32) - pk)
                    s = s + dv[:, None] * kf[None, :]
                    dst = state_idx[n, i]
                    if dst > 0:
                        state[dst, h, g, :, :] = s
                    o = pq + dv * kq
                    rstd = torch.rsqrt(torch.mean(o * o, dim=-1) + eps)
                    zf = z[row, h, g, :].to(torch.float32)
                    if act == 0:
                        gate = zf * torch.sigmoid(zf)
                    else:
                        gate = torch.sigmoid(zf)
                    out[row, h, g, :] = (o * rstd * norm_w[:] * gate).to(out.dtype)
        else:
            for i in hl.static_range(W):
                if i < tn:
                    out[bos + i, h, g, :] = hl.zeros([V], dtype=out.dtype)
    return out, state


def _gated_rms(o, zf, norm_w, eps, act):
    """bf16-bound gated RMS norm of one f32 [V] row (helper inlined into the bodies)."""
    rstd = torch.rsqrt(torch.mean(o * o, dim=-1) + eps)
    if act == 0:
        gate = zf * torch.sigmoid(zf)
    else:
        gate = torch.sigmoid(zf)
    return o * rstd * norm_w * gate


def kgdn1_vt_norm(o, z, norm_w, out, eps: float, act: hl.constexpr):
    """Decode norm launch: o f32 [T, H, G, V] -> out [T, H, G, V] bf16."""
    T, H, G, V = o.shape
    V = hl.specialize(V)
    for t, h, g in hl.grid([T, H, G]):
        out[t, h, g, :] = _gated_rms(o[t, h, g, :], z[t, h, g, :].to(torch.float32),
                                     norm_w[:], eps, act).to(out.dtype)
    return out


def kgdn1_vt_norm_seq(o, z, norm_w, out, cu_lo, cu_hi, eps: float, act: hl.constexpr):
    """Verify norm launch: o f32 [N, W, H, G, V] (sequence-major) -> out rows
    cu_lo[n] + i for i < min(W, tokens of n); every other row untouched."""
    N, W, H, G, V = o.shape
    W = hl.specialize(W)
    V = hl.specialize(V)
    for n, i, h, g in hl.grid([N, W, H, G]):
        if i < cu_hi[n] - cu_lo[n]:
            row = cu_lo[n] + i
            out[row, h, g, :] = _gated_rms(o[n, i, h, g, :], z[row, h, g, :].to(torch.float32),
                                           norm_w[:], eps, act).to(out.dtype)
    return out


_NORM_KERNELS: dict = {}


def _norm_kernel(body):
    """The norm launch at its fixed config (compiled once per process)."""
    kern = _NORM_KERNELS.get(body)
    if kern is None:
        kern = _NORM_KERNELS[body] = helion.kernel(
            body, config=helion.Config(**NORM_CONFIG), static_shapes=False)
    return kern


def vt_post(res, q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, out, scale, eps,
            act):
    """kgdn1_decode_vt's second launch: (o, state) -> (out, state)."""
    o, state = res
    _norm_kernel(kgdn1_vt_norm)(o, z, norm_w, out, eps, act)
    return out, state


def verify_vt_post(res, q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, cu_lo,
                   cu_hi, acc, out, scale, eps, act):
    """kgdn1_verify_vt's second launch: (o, state) -> (out, state)."""
    o, state = res
    _norm_kernel(kgdn1_vt_norm_seq)(o, z, norm_w, out, cu_lo, cu_hi, eps, act)
    return out, state


def kgdn1_decode_vt(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, out,
                    scale: float, eps: float, act: hl.constexpr):
    """kgdn1_decode's args, V-tiled: grid (t, h, g, V tile). Updates the state
    in place and returns (o f32 [T, H, G, V] unnormalized, state); z, norm_w,
    out, eps, act are vt_post's (unused here). NULL slot: o row 0 (-> out 0)."""
    T, H, K = q.shape
    G, V = v.shape[2], v.shape[3]
    K = hl.specialize(K)
    V = hl.specialize(V)
    bv = hl.register_tunable("v_block", PowerOfTwoFragment(16, 64, 32))
    o = torch.empty([T, H, G, V], dtype=torch.float32, device=q.device)
    for tt, th, tg, tv in hl.tile([T, H, G, V], block_size=[1, 1, 1, bv]):
        t, h, g = tt.begin, th.begin, tg.begin
        slot = state_idx[t]
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
            s = state[slot, h, g, tv, :] * decay
            pk = torch.sum(s * kf[None, :], dim=-1)
            pq = torch.sum(s * qf[None, :], dim=-1)
            dv = beta * (v[t, h, g, tv].to(torch.float32) - pk)
            state[slot, h, g, tv, :] = s + dv[:, None] * kf[None, :]
            o[t, h, g, tv] = pq + dv * kq
        else:
            o[t, h, g, tv] = hl.zeros([tv], dtype=torch.float32)  # norm(0) = 0: zero row
    return o, state


def kgdn1_verify_vt(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, cu_lo, cu_hi,
                    acc, out, scale: float, eps: float, act: hl.constexpr):
    """kgdn1_verify's args, V-tiled: grid (n, h, g, V tile); each program runs
    the W-token recurrence on its V rows of the state. Returns (o f32
    [N, W, H, G, V] unnormalized, sequence-major; state); verify_vt_post norms."""
    T, H, K = q.shape
    G, V = v.shape[2], v.shape[3]
    N, W = state_idx.shape
    K = hl.specialize(K)
    V = hl.specialize(V)
    W = hl.specialize(W)
    bv = hl.register_tunable("v_block", PowerOfTwoFragment(16, 64, 32))
    o = torch.empty([N, W, H, G, V], dtype=torch.float32, device=q.device)
    for tn_, th, tg, tv in hl.tile([N, H, G, V], block_size=[1, 1, 1, bv]):
        n, h, g = tn_.begin, th.begin, tg.begin
        bos = cu_lo[n]
        tn = cu_hi[n] - bos
        na = acc[n]
        src = state_idx[n, torch.clamp(na - 1, 0, W - 1)]
        if (src > 0) & (na >= 1) & (na <= W):
            s = state[src, h, g, tv, :]
            neg_a = -torch.exp(a_log[h, g])
            for i in hl.static_range(W):
                if i < tn:
                    row = bos + i
                    qf = q[row, h, :].to(torch.float32)
                    kf = k[row, h, :].to(torch.float32)
                    qf = qf * (scale / torch.sqrt(torch.sum(qf * qf, dim=-1) + 1e-6))
                    kf = kf / torch.sqrt(torch.sum(kf * kf, dim=-1) + 1e-6)
                    kq = torch.sum(kf * qf, dim=-1)
                    x = a[row, h, g].to(torch.float32) + dt_bias[h, g]
                    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
                    beta = torch.sigmoid(b[row, h, g].to(torch.float32))
                    s = s * torch.exp(neg_a * sp)
                    pk = torch.sum(s * kf[None, :], dim=-1)
                    pq = torch.sum(s * qf[None, :], dim=-1)
                    dv = beta * (v[row, h, g, tv].to(torch.float32) - pk)
                    s = s + dv[:, None] * kf[None, :]
                    dst = state_idx[n, i]
                    if dst > 0:
                        state[dst, h, g, tv, :] = s
                    o[n, i, h, g, tv] = pq + dv * kq
                else:  # never read by the norm; written so no output stays torch.empty
                    o[n, i, h, g, tv] = hl.zeros([tv], dtype=torch.float32)
        else:
            for i in hl.static_range(W):
                o[n, i, h, g, tv] = hl.zeros([tv], dtype=torch.float32)
    return o, state


def body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads):
    """The oxide op's tensors as the body's head-split views (zero copies):
    mixed_qkv [T, 2HK+HV*V] -> q, k, v; ba [T, >=2HV] -> b, a; HV -> (H, G).
    z is [T, HV, V] (flat layout) or already [T, H, G, V] (the de-interleaved
    Qwen3-Next view, qwen_gdn.deinterleave)."""
    T = mixed_qkv.shape[0]
    S, HV, V, K = state.shape
    H = num_k_heads
    G = HV // H
    q = mixed_qkv[:, :H * K].unflatten(1, (H, K))
    k = mixed_qkv[:, H * K:2 * H * K].unflatten(1, (H, K))
    v = mixed_qkv[:, 2 * H * K:2 * H * K + HV * V].unflatten(1, (H, G, V))
    z = z if z.dim() == 4 else z.unflatten(1, (H, G))
    return (q, k, v, ba[:, :HV].unflatten(1, (H, G)), ba[:, HV:2 * HV].unflatten(1, (H, G)),
            z, a_log.view(H, G), dt_bias.view(H, G), norm_w, state.unflatten(1, (H, G)),
            state_idx, out.view(T, H, G, V))


def verify_body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, cu, acc, out,
                     num_k_heads):
    """kgdn1_verify's args from the lane's verify tensors (zero copies)."""
    *head, idx, o = body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out,
                              num_k_heads)
    return (*head, idx, cu[:-1], cu[1:], acc, o)


def case(numtokens, h, hv, k, v) -> dict:
    return {"numtokens": numtokens, "h": h, "hv": hv, "k": k, "v": v}


def verify_case(nseq, h, hv, k, v, w) -> dict:
    return dict(case(nseq * w, h, hv, k, v), w=w)


# ---------------------------------------------------------------------------
# serving (the lane: suffix_hybrid/kernels/qwen_gdn.py)
# ---------------------------------------------------------------------------
def _check_pool(state):
    # Helion indexes in int32 here (dynamic shapes size the choice by numel,
    # not by vLLM's padded page): refuse a pool whose slot offsets overflow.
    S, HV, V, K = state.shape
    if (S - 1) * state.stride(0) + HV * V * K > INT32_MAX:
        raise RuntimeError(f"{hk.MARKER} {KERNEL}: state pool spans > 2^31 elements "
                           f"(S={S}, page {state.stride(0)}); run {ENV}=oxide")


DECODE_BODIES = ((KERNEL, kgdn1_decode), (KERNEL_VT, kgdn1_decode_vt))  # pick order
POSTS = {KERNEL_VT: vt_post, VERIFY_KERNEL_VT: verify_vt_post}
VERIFY_BODIES = ((VERIFY_KERNEL, kgdn1_verify), (VERIFY_KERNEL_VT, kgdn1_verify_vt))


def pick(bodies, c):
    """The kernel name serving case `c`: the first of `bodies` (full tile, then
    V-tiled) with a shipped config; None = the baseline serves."""
    return next((name for name, _ in bodies if hk.config(name, c) is not None), None)


def _compiled(bodies, c):
    name = pick(bodies, c)
    if name is None:
        raise KeyError(f"{hk.MARKER} no tuned {' / '.join(n for n, _ in bodies)} config for "
                       f"{hk.case_key(c)} on {hk.platform()}")
    kern = hk.compiled(dict(bodies)[name], name, c)
    post = POSTS.get(name)
    return kern if post is None else (lambda *args: post(kern(*args), *args))


def gdn_decode_fused_helion(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx,
                            out, num_k_heads, scale, norm_eps, act):
    """One K-GDN1 step on Helion (the bucket's picked body; KeyError if none)."""
    T = mixed_qkv.shape[0]
    if T == 0:
        return None
    _check_pool(state)
    S, HV, V, K = state.shape
    kern = _compiled(DECODE_BODIES, case(hk.bucket(T, BUCKETS), num_k_heads, HV, K, V))
    kern(*body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out,
                    num_k_heads), float(scale), float(norm_eps), int(act))
    return None


def gdn_verify_fused_helion(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, cu,
                            acc, out, num_k_heads, scale, norm_eps, act):
    """One MTP verify step on Helion: state_idx [N, W], cu [N+1], acc [N]."""
    N, W = state_idx.shape
    if N == 0:
        return None
    _check_pool(state)
    S, HV, V, K = state.shape
    kern = _compiled(VERIFY_BODIES, verify_case(hk.bucket(N, BUCKETS), num_k_heads, HV, K, V, W))
    kern(*verify_body_args(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, cu, acc,
                           out, num_k_heads), float(scale), float(norm_eps), int(act))
    return None


def _missing(kernel, c, what):
    return RuntimeError(
        f"{ENV}=helion but no tuned {kernel} config (nor {kernel}_vt) for {hk.case_key(c)} on "
        f"{hk.platform()} "
        f"(tune it, ship configs/helion/{kernel}/<platform>.json, or use {ENV}=auto"
        f"{what})")


def oxide_fits(h, hv, k, v, layout="flat") -> bool:
    """The shipped oxide cubin runs exactly this per-rank shape and layout."""
    return (h, hv, k) == OXIDE_SHAPE and k == v and layout == "flat"


def decode_op(native, h, hv, k, v, layout="flat"):
    """(op, helion_buckets) for one layer shape under ENV (helion | auto).

    op has the oxide op's signature (``native.gdn_decode_fused_cuda``, whose
    stream_ptr is torch's current stream — the stream Helion launches on).
    helion: every bucket needs a shipped config, else RuntimeError now (load).
    auto: Helion where a config is shipped (= it beat the baseline in the
    lab), elsewhere the baseline: the oxide op where its cubin fits the
    shape/layout, vLLM's stock FLA decode + layer_norm_fwd chain otherwise."""
    mode = hk.backend(ENV)
    use = {}
    for m in BUCKETS:
        c = case(m, h, hv, k, v)
        use[m] = pick(DECODE_BODIES, c) is not None
        if mode == "helion" and not use[m]:
            raise _missing(KERNEL, c, "|oxide" if oxide_fits(h, hv, k, v, layout) else "")
    if oxide_fits(h, hv, k, v, layout):
        fallback = native.gdn_decode_fused_cuda
    else:
        def fallback(*args):  # stream_ptr dropped: stock runs on torch's current stream
            return qwen_gdn.stock_decode_fused(*args[:-1])
    buckets = tuple(m for m in BUCKETS if use[m])
    if not buckets:
        return fallback, ()

    def op(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads,
           scale, norm_eps, act, stream_ptr):
        if use[hk.bucket(mixed_qkv.shape[0], BUCKETS)]:
            return gdn_decode_fused_helion(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state,
                                           state_idx, out, num_k_heads, scale, norm_eps, act)
        return fallback(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out,
                        num_k_heads, scale, norm_eps, act, stream_ptr)

    return op, buckets


def served(h, hv, k, v, w=0) -> dict:
    """{bucket: kernel name} of the Helion body each bucket serves on (K5 log);
    w >= 2: the verify buckets (sequence counts), else decode."""
    if w >= 2:
        return {m: n for m in BUCKETS
                if (n := pick(VERIFY_BODIES, verify_case(m, h, hv, k, v, w)))}
    return {m: n for m in BUCKETS if (n := pick(DECODE_BODIES, case(m, h, hv, k, v)))}


def verify_op(h, hv, k, v, w):
    """(op | None, helion_buckets) for the MTP verify step (w = k + 1 tokens).
    Buckets are sequence counts. helion: every bucket needs a config, else
    RuntimeError now. auto: Helion where shipped; the lane keeps every other
    bucket on vLLM's own spec path (its CUDA MTP kernel or FLA + norm)."""
    mode = hk.backend(ENV)
    if mode == "oxide" or w < 2:
        return None, ()
    buckets = []
    for m in BUCKETS:
        c = verify_case(m, h, hv, k, v, w)
        if pick(VERIFY_BODIES, c) is not None:
            buckets.append(m)
        elif mode == "helion":
            raise _missing(VERIFY_KERNEL, c, " (vLLM's MTP kernel on unshipped buckets)")
    return (gdn_verify_fused_helion if buckets else None), tuple(buckets)


# ---------------------------------------------------------------------------
# autotune lab (inference-console api/autotune_kernels.py ABI)
# ---------------------------------------------------------------------------
def gdn_dims(model):
    """(H, HV, K, V) of the FULL model: config.json's linear_* fields when the
    caller passes them, else the GDN_DIMS table; None for non-GDN models."""
    vals = [model.get(f) for f in LINEAR_FIELDS]
    if all(isinstance(x, int) and x > 0 for x in vals):
        return tuple(vals)
    return GDN_DIMS.get(model.get("architecture"), {}).get(model.get("hidden"))


def _rank_shapes(model):
    """Per-rank (h, hv, k, v) for the model's tp (every TP in TPS without one);
    a TP whose heads do not divide is skipped."""
    dims = gdn_dims(model)
    if dims is None:
        return []
    H, HV, K, V = dims
    tps = (model["tp"],) if model.get("tp") else TPS
    return [(H // tp, HV // tp, K, V) for tp in tps if not H % tp and not HV % tp]


def cases(model) -> list:
    """Per-TP-rank GDN head layout x numtokens buckets; [] for non-GDN models."""
    return [case(m, *s) for s in _rank_shapes(model) for m in BUCKETS]


def verify_cases(model) -> list:
    """Verify cases for the model's MTP k (spec_k): sequence buckets x w = k+1."""
    w = (model.get("spec_k") or 0) + 1
    if w < 2:
        return []
    return [verify_case(m, *s, w) for s in _rank_shapes(model) for m in BUCKETS]


def problem(c, seed, device="cuda"):
    """The oxide op's args at case `c`: the lane's make_inputs (bf16 row-strided
    qkvz views, fp32 state in vLLM's padded pages), distinct live slots in
    random order, act = silu (every Qwen GDN checkpoint)."""
    T, H, HV, K, V = c["numtokens"], c["h"], c["hv"], c["k"], c["v"]
    slots = T + 1
    gen = torch.Generator().manual_seed(7919 + seed)
    idx = (torch.randperm(slots - 1, generator=gen)[:T] + 1).tolist()
    inp = qwen_gdn.make_inputs(T, H, HV, K, slots, device=device, seed=seed, idx=idx,
                               page_pad=4096, V=V)
    out = torch.zeros(T, HV, V, dtype=torch.bfloat16, device=device)
    return (inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], out, H, K ** -0.5, 1e-6, 0)


def make(c, seed, device="cuda"):
    *tensors, num_k_heads, scale, eps, act = problem(c, seed, device)
    return (*body_args(*tensors, num_k_heads), scale, eps, act)


def make_baseline(c, seed):
    args = problem(c, seed)
    H, HV, K, V = args[9], args[6].shape[1], args[6].shape[3], args[6].shape[2]
    if oxide_fits(H, HV, K, V):
        qwen_gdn.load_prebuilt(H, HV, K)  # driver-load the cubin outside graph capture
    return args


def baseline(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads,
             scale, eps, act):
    """The oxide op where its cubin's shape fits, else vLLM's stock chain."""
    if oxide_fits(num_k_heads, state.shape[1], state.shape[3], state.shape[2]):
        qwen_gdn._native().gdn_decode_fused_cuda(
            mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx, out, num_k_heads,
            scale, eps, act, torch.cuda.current_stream().cuda_stream)
    else:
        qwen_gdn.stock_decode_fused(mixed_qkv, z, ba, a_log, dt_bias, norm_w, state,
                                    state_idx, out, num_k_heads, scale, eps, act)
    return out, state


def verify_problem(c, seed, device="cuda"):
    """The lane's verify args at case `c`: numtokens // w live sequences of w
    tokens, random accepted counts, distinct slots (qwen_gdn.make_verify_inputs)."""
    w = c["w"]
    n = c["numtokens"] // w
    inp = qwen_gdn.make_verify_inputs(n, w, c["h"], c["hv"], c["k"], V=c["v"], device=device,
                                      seed=seed, lens=[w] * n, page_pad=4096)
    out = torch.zeros(n * w, c["hv"], c["v"], dtype=torch.bfloat16, device=device)
    return (inp["mixed_qkv"], inp["z"], inp["ba"], inp["a_log"], inp["dt_bias"],
            inp["norm_w"], inp["state"], inp["state_idx"], inp["cu"], inp["acc"], out, c["h"],
            c["k"] ** -0.5, 1e-6, 0)


def verify_make(c, seed, device="cuda"):
    *tensors, num_k_heads, scale, eps, act = verify_problem(c, seed, device)
    return (*verify_body_args(*tensors, num_k_heads), scale, eps, act)


def verify_make_baseline(c, seed):
    return verify_problem(c, seed)


def verify_baseline(*args):
    """vLLM's own verify step (qwen_gdn.stock_verify_fused)."""
    qwen_gdn.stock_verify_fused(*args)
    return args[10], args[6]


def _recur(s, qf, kf, v, b, a, z, a_log, dt_bias, norm_w, scale, eps, act):
    """One recurrence step, broadcast over leading dims: s [..., H, G, V, K]
    f32, qf/kf [..., H, K] raw, v/z [..., H, G, V], b/a [..., H, G].
    Returns (new s, y f32 [..., H, G, V])."""
    qf = qf.float() * (scale / torch.sqrt((qf.float() ** 2).sum(-1, keepdim=True) + 1e-6))
    kf = kf.float() / torch.sqrt((kf.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    x = a.float() + dt_bias
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    decay = torch.exp(-torch.exp(a_log) * sp)
    beta = torch.sigmoid(b.float())
    kk, qq = kf[..., None, None, :], qf[..., None, None, :]       # [..., H, 1, 1, K]
    s = s * decay[..., None, None]
    dv = beta[..., None] * (v.float() - (s * kk).sum(-1))
    s = s + dv[..., None] * kk
    o = (s * qq).sum(-1)                                            # [..., H, G, V]
    zf = z.float()
    gate = zf * torch.sigmoid(zf) if act == 0 else torch.sigmoid(zf)
    return s, o * torch.rsqrt((o * o).mean(-1, keepdim=True) + eps) * norm_w * gate


def reference(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, out, scale, eps,
              act):
    """Exact eager answer (vLLM semantics = the oxide twin's, f32 o into the
    norm), vectorized over tokens and heads. Does not mutate its args."""
    idx = state_idx.long()
    live = idx > 0
    rows = idx[live]
    s, y = _recur(state[rows].float(), q[live], k[live], v[live], b[live], a[live], z[live],
                  a_log, dt_bias, norm_w, scale, eps, act)
    new_state = state.clone()
    new_state[rows] = s
    ref_out = torch.zeros(out.shape, dtype=out.dtype, device=out.device)
    ref_out[live] = y.to(out.dtype)
    return ref_out, new_state


def verify_reference(q, k, v, b, a, z, a_log, dt_bias, norm_w, state, state_idx, cu_lo, cu_hi,
                     acc, out, scale, eps, act):
    """Exact eager verify (vLLM's spec semantics), sequential per sequence.
    Rows outside every sequence keep `out`'s values. Does not mutate its args."""
    N, W = state_idx.shape
    new_state = state.clone()
    ref_out = out.clone()
    for n in range(N):
        bos, tn, na = int(cu_lo[n]), int(cu_hi[n]) - int(cu_lo[n]), int(acc[n])
        src = int(state_idx[n, na - 1]) if 1 <= na <= W else 0
        if src <= 0:
            ref_out[bos:bos + min(tn, W)] = 0
            continue
        s = state[src].float()
        for i in range(min(tn, W)):
            r = bos + i
            s, y = _recur(s, q[r], k[r], v[r], b[r], a[r], z[r], a_log, dt_bias, norm_w, scale,
                          eps, act)
            dst = int(state_idx[n, i])
            if dst > 0:
                new_state[dst] = s
            ref_out[r] = y.to(out.dtype)
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
    {"name": VERIFY_KERNEL, "body": kgdn1_verify, "cases": verify_cases, "make": verify_make,
     "make_baseline": verify_make_baseline, "reference": verify_reference,
     "baseline": verify_baseline,
     "baseline_name": ("vLLM fused_gdn_decode_post_conv_mtp (K=V=128, HV/H in 1,2,3,4,8, "
                       "w <= 8); FLA fused_sigmoid_gating_delta_rule_update + layer_norm_fwd "
                       "otherwise"),
     "check": check,
     "replaces": [r"gdn_decode_post_conv_mtp_kernel",
                  r"fused_sigmoid_gating_delta_rule_update_kernel", r"^layer_norm_fwd_kernel"]},
]
# V-tiled bodies: same cases, reference, baseline and check; tuned and timed
# next to the full-tile ones; `post` (helion_tune: checked and timed as body +
# post, only the body autotuned) adds the norm launch.
KERNELS += [dict(KERNELS[0], name=KERNEL_VT, body=kgdn1_decode_vt, post=vt_post),
            dict(KERNELS[1], name=VERIFY_KERNEL_VT, body=kgdn1_verify_vt, post=verify_vt_post)]
