# SPDX-License-Identifier: Apache-2.0
"""MoE top-k gating as one Triton program per token (SUFFIX_ROCM_TOPK_GATING=1, ROCm).

vLLM's ROCm MoE router calls aiter.topk_softmax (shared expert fused as a sigmoid-scored
extra column, VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1): AITER's topkGatingSoftmax
spreads a 512-expert row over 16 threads and runs 10 dependent argmax rounds with
butterfly reductions; on the MI350P that is ~12 us per MoE layer at c8..c32 (50 layers
per MTP-4 step: 0.6 ms, ~3% of a c8 step) for a few KB of logits.

Same contract: topk_indices [T, k] get the k largest routed logits in descending order
(ties to the lower index), topk_weights[:, :k] = exp(l_i - l_max) renormalized over the
k (renormalize=True) or over the whole row, topk_weights[:, k + s] = sigmoid(shared
logit s), token_expert_indices[t, j] = j * T + t. Indices are bit-identical; weights use
libdevice exp (HIP's expf) in AITER's summation order.

    python -m suffix_hybrid.kernels.topk_gating_rocm   # GPU oracle vs aiter.topk_softmax + us/call
"""
from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

MARK = "[suffix topk-gating]"
# HIP's expf (ocml) when this Triton exposes it, so the weights follow AITER's math.
_LIBDEVICE = hasattr(getattr(tl, "extra", None), "hip")


@triton.jit
def _exp(x, LIBDEVICE: tl.constexpr):
    if LIBDEVICE:
        return tl.extra.hip.libdevice.exp(x)
    return tl.exp(x)


@triton.jit(do_not_specialize=["T"])
def _topk_gating_kernel(
    logits_ptr, w_ptr, idx_ptr, src_ptr, T, stride_in, stride_w, stride_idx,
    E: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, NS: tl.constexpr,
    RENORM: tl.constexpr, BLOCK_E: tl.constexpr, LIBDEVICE: tl.constexpr,
):
    NSP: tl.constexpr = triton.next_power_of_2(NS)
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    x = tl.load(logits_ptr + row * stride_in + offs, mask=offs < E,
                other=-float("inf")).to(tl.float32)
    m0 = tl.max(x, axis=0)
    kk = tl.arange(0, KP)
    nums = tl.zeros([KP], dtype=tl.float32)
    ids = tl.zeros([KP], dtype=tl.int32)
    renorm = 0.0
    for k_idx in tl.static_range(K):
        v = tl.max(x, axis=0)
        e = tl.min(tl.where(x == v, offs, BLOCK_E), axis=0)  # ties: the lower index
        numer = _exp(v - m0, LIBDEVICE)
        nums = tl.where(kk == k_idx, numer, nums)
        ids = tl.where(kk == k_idx, e, ids)
        renorm += numer  # AITER's order: k_idx ascending
        x = tl.where(offs == e, -float("inf"), x)
    if RENORM:
        scale = tl.where(renorm != 0.0, 1 / renorm, 1.0)
    else:
        rest = tl.sum(tl.where(x > -float("inf"), _exp(x - m0, LIBDEVICE), 0.0), axis=0)
        z = renorm + rest
        scale = tl.where(z != 0.0, 1.0 / z, 1.0)
    tl.store(w_ptr + row * stride_w + kk, nums * scale, mask=kk < K)
    tl.store(idx_ptr + row * stride_idx + kk, ids, mask=kk < K)
    tl.store(src_ptr + row * K + kk, kk * T + row, mask=kk < K)
    if NS > 0:
        s = tl.arange(0, NSP)
        logit = tl.load(logits_ptr + row * stride_in + E + s, mask=s < NS).to(tl.float32)
        tl.store(w_ptr + row * stride_w + K + s, 1.0 / (1.0 + _exp(-logit, LIBDEVICE)),
                 mask=s < NS)


def supported(topk_weights, topk_indices, token_expert_indices, gating_output,
              num_shared_experts, scoring) -> bool:
    e = gating_output.shape[-1] - num_shared_experts
    return (gating_output.dim() == 2 and gating_output.stride(1) == 1
            and gating_output.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and topk_weights.dtype == torch.float32 and topk_weights.stride(1) == 1
            and topk_indices.dtype == torch.int32 and topk_indices.stride(1) == 1
            and token_expert_indices.dtype == torch.int32 and token_expert_indices.is_contiguous()
            and 0 < e <= 1024 and topk_indices.shape[1] <= 32
            and num_shared_experts in (0, 1, 2) and (num_shared_experts == 0 or scoring == "sigmoid"))


def topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output, renormalize,
                 num_shared_experts=0, shared_expert_scoring_func=""):
    """aiter.topk_softmax's signature; shapes AITER does not take here go to AITER."""
    if not supported(topk_weights, topk_indices, token_expert_indices, gating_output,
                     num_shared_experts, shared_expert_scoring_func):
        from aiter import topk_softmax as aiter_topk_softmax
        return aiter_topk_softmax(topk_weights, topk_indices, token_expert_indices,
                                  gating_output, renormalize, num_shared_experts,
                                  shared_expert_scoring_func)
    t, k = gating_output.shape[0], topk_indices.shape[1]
    e = gating_output.shape[1] - num_shared_experts
    if t:
        _topk_gating_kernel[(t,)](
            gating_output, topk_weights, topk_indices, token_expert_indices, t,
            gating_output.stride(0), topk_weights.stride(0), topk_indices.stride(0),
            E=e, K=k, KP=triton.next_power_of_2(k), NS=num_shared_experts,
            RENORM=bool(renormalize), BLOCK_E=triton.next_power_of_2(e), LIBDEVICE=_LIBDEVICE,
            num_warps=1 if e <= 256 else 2)
    return None


def main() -> int:
    """Oracle on silicon vs aiter.topk_softmax at Qwen3.8-Flash-Next's router (512 routed
    experts + 1 fused shared expert, top-10, bf16 logits): indices and token_expert_indices
    bitwise, weights within fp32 rounding; graphed us/call over 50 calls (one per MoE layer)."""
    from aiter import topk_softmax as aiter_topk_softmax

    from suffix_hybrid.kernels.hc_fused_rocm import _graph_us

    dev = "cuda"
    torch.manual_seed(0)
    failed = False
    for t, e, k, ns, renorm in ((1, 512, 10, 1, True), (5, 512, 10, 1, True),
                                (40, 512, 10, 1, True), (160, 512, 10, 1, True),
                                (512, 512, 10, 1, True), (40, 512, 10, 0, True),
                                (40, 512, 10, 1, False), (37, 128, 8, 0, True)):
        logits = torch.randn(t, e + ns, device=dev).to(torch.bfloat16)
        logits[: t // 2, 3] = logits[: t // 2, 7]  # ties: lower index first
        outs = []
        for fn in (aiter_topk_softmax, topk_softmax):
            w = torch.full((t, k + ns), float("nan"), device=dev)
            i = torch.full((t, k), -1, dtype=torch.int32, device=dev)
            s = torch.full((t, k), -1, dtype=torch.int32, device=dev)
            fn(w, i, s, logits, renorm, ns, "sigmoid" if ns else "")
            outs.append((w, i, s))
        (w0, i0, s0), (w1, i1, s1) = outs
        ok = torch.equal(i0, i1) and torch.equal(s0, s1) and torch.allclose(w0, w1, rtol=2e-6,
                                                                              atol=1e-9)
        failed |= not ok
        line = (f"{MARK} T={t} E={e} k={k} shared={ns} renorm={renorm}: "
                f"{'MATCH' if ok else 'MISMATCH'} (indices {'bitwise' if torch.equal(i0, i1) else 'DIFFER'}, "
                f"weights max abs diff {(w0 - w1).abs().max().item():.2e}, bit-exact "
                f"{100 * (w0 == w1).float().mean().item():.1f}%)")
        if e == 512 and ns == 1 and renorm:
            ws = [torch.empty(t, k + ns, device=dev) for _ in range(50)]
            idx = [torch.empty(t, k, dtype=torch.int32, device=dev) for _ in range(50)]
            src = [torch.empty(t, k, dtype=torch.int32, device=dev) for _ in range(50)]
            lg = [torch.randn(t, e + ns, device=dev).to(torch.bfloat16) for _ in range(50)]
            t_a = _graph_us(lambda j: aiter_topk_softmax(ws[j], idx[j], src[j], lg[j], True, 1,
                                                         "sigmoid"), 50)[0]
            t_n = _graph_us(lambda j: topk_softmax(ws[j], idx[j], src[j], lg[j], True, 1,
                                                   "sigmoid"), 50)[0]
            line += f" | graphed AITER {t_a:.1f} us -> {t_n:.1f} us"
        print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
