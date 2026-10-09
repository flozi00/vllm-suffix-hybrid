# SPDX-License-Identifier: Apache-2.0
"""GDN MTP-verify core on ROCm with AITER's strided gated delta rule (SUFFIX_ROCM_GDN_MTP=1).

With MTP every decode step of vLLM 81198e97 takes the generic spec branch of
QwenGatedDeltaNetAttention._forward_core_rocm: zero_, z copy, conv1d update,
3 q/k/v copies + cat, 2 a/b copies, FLA fused_sigmoid_gating_delta_rule_update,
output copy = 11 launches per GDN layer, 2 of them math. forward_spec keeps the
same conv1d call and runs AITER's fused_rearrange_sigmoid_gated_delta_rule
straight on the packed qkv, writing core_attn_out in place: conv, one a/b copy
and the delta rule = 3 launches per layer, plus one index shift per forward.
The patched forward_hip feeds the output gate the z columns of qkvz (strided
view): no path writes them, so it reads exactly what the z_out copy held.

State contract (spec-decode rollback): the AITER Triton kernel is FLA's kernel
line for line (read slot idx[n, accepted-1], write the state after token t to
idx[n, t], same math order, same grid) except the slot sentinel: FLA skips
slots <= 0 (NULL_BLOCK_ID = 0), AITER skips < 0. Dummy and capture runs zero
the block table on purpose, so the difference is reachable. Indices - 1 against
ssm_state[1:] address the same slots and skip the same ones. Rows vLLM zeroes
past num_actual_tokens are zeroed here too; rows of zero-length or skipped
sequences are unwritten in both (stock copies them from an uninitialized
buffer). Never FlyDSL: no draft_window, so capture-safe Triton only.

    python -m suffix_hybrid.kernels.gdn_mtp_rocm   # GPU oracle vs vLLM + us/call (boot gate gdn_mtp_bench)
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

_Q = None  # vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn, set by install()


def install(module) -> None:
    """rocm_patches `after` hook: the patched _forward_core_rocm calls _suffix_gdn_mtp first."""
    global _Q
    _Q = module
    module._suffix_gdn_mtp = forward_spec


def _state_indices_m1(md):
    # One launch per forward pass (FULL capture records it in the first GDN
    # layer), shared by every layer that sees this metadata. The cache holds
    # the source tensor, so its id cannot be reused while cached.
    cache = vars(_Q.get_forward_context()).setdefault("_suffix_gdn_mtp_idx", {})
    src = md.spec_state_indices_tensor
    hit = cache.get(id(src))
    if hit is None:
        hit = cache[id(src)] = (src, src - 1)
    return hit[1]


def forward_spec(layer, qkvz, ba, core_attn_out, md) -> bool:
    """_forward_core_rocm for an all-spec (MTP verify) batch; False = not handled."""
    if (md.spec_sequence_masks is None or md.num_prefills or md.num_decodes
            or layer.gqa_interleaved_layout):
        return False
    n = md.num_actual_tokens
    key_dim, value_dim = layer.key_dim // layer.tp_size, layer.value_dim // layer.tp_size
    idx = md.spec_state_indices_tensor
    conv_state = layer.kv_cache[0]
    if not _Q.is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    w = layer.conv1d.weight
    # vLLM's spec conv call, unchanged: it overwrites the qkv columns of qkvz in place.
    qkv = _Q.causal_conv1d_update(
        qkvz[:n, : 2 * key_dim + value_dim], conv_state, w.view(w.size(0), w.size(2)),
        layer.conv1d.bias, layer.activation,
        conv_state_indices=idx[:, 0][: md.num_spec_decodes],
        num_accepted_tokens=md.num_accepted_tokens,
        query_start_loc=md.spec_query_start_loc,
        max_query_len=idx.size(-1),
        validate_data=False,
    )
    # ponytail: one a/b copy left (the kernel assumes row stride HV); a delta-rule
    # kernel taking the ba row stride drops it (fusion plan F1b).
    b, a = ba[:n].unflatten(-1, (2, value_dim // layer.head_v_dim)).transpose(0, 1).contiguous()
    _Q.gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
        A_log=layer.A_log, a=a, b=b, dt_bias=layer.dt_bias, qkv=qkv,
        key_dim=key_dim, value_dim=value_dim,
        head_k_dim=layer.head_k_dim, head_v_dim=layer.head_v_dim,
        initial_state=layer.kv_cache[1][1:], inplace_final_state=True,
        cu_seqlens=md.spec_query_start_loc[: md.num_spec_decodes + 1],
        ssm_state_indices=_state_indices_m1(md),
        num_accepted_tokens=md.num_accepted_tokens,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=core_attn_out,
    )
    if n < core_attn_out.shape[0]:
        core_attn_out[n:].zero_()  # stock zeroes everything, then overwrites rows < n
    return True


def main() -> int:
    """Oracle on silicon: vLLM's own _forward_core_rocm spec branch vs forward_spec on
    Qwen3.8-Flash-Next GDN shapes (16 qk / 48 v heads x 128, conv 4, MTP-4, fp32 state).
    Compares every state page byte, the defined output rows, the zeroed tail and the z
    the output gate reads; then replays a captured graph with new indices and inputs."""
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as Q
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from suffix_hybrid.kernels.qsa_mqa_rocm import _time_us

    if not Q.GDN_AITER_TRITON_AVAILABLE:
        print("[suffix gdn-mtp] AITER GDN Triton kernels not importable "
              "(VLLM_ROCM_USE_AITER=1?): not run", flush=True)
        return 2
    install(Q)
    Q._suffix_gdn_mtp = lambda *a: False  # a patched module's own branch falls through: stock

    dev, bf16 = "cuda", torch.bfloat16
    hk, hv, dk, dv, width, spec = 16, 48, 128, 128, 4, 4
    win, qkv_dim = spec + 1, 2 * hk * dk + hv * dv
    torch.manual_seed(0)
    layer = Q.QwenGatedDeltaNetAttention.__new__(Q.QwenGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.__dict__.update(
        prefix="gdn", tp_size=1, num_k_heads=hk, num_v_heads=hv, head_k_dim=dk, head_v_dim=dv,
        key_dim=hk * dk, value_dim=hv * dv, gqa_interleaved_layout=False, activation="silu",
        enable_packed_recurrent_decode=False,
        conv1d=SimpleNamespace(weight=0.3 * torch.randn(qkv_dim, 1, width, device=dev, dtype=bf16),
                               bias=None),
        A_log=0.5 * torch.randn(hv, device=dev),
        dt_bias=0.5 * torch.randn(hv, device=dev, dtype=bf16))
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, hk, hv, dk, dv, width, spec)
    cbytes, sbytes = math.prod(conv_shape) * 2, math.prod(ssm_shape) * 4
    page = cbytes + sbytes + 4096  # padded page, like the hybrid allocator's

    def views(pages):
        return (pages[:, :cbytes].view(bf16).view(-1, *conv_shape),
                pages[:, cbytes:cbytes + sbytes].view(torch.float32).view(-1, *ssm_shape))

    def case(real, pad, tail, acc, null_slots=False):
        seqs, tokens = real + pad, real * win
        n = seqs * win if pad else tokens  # FULL graph: num_actual_tokens = padded tokens
        pool = 2 * seqs * win + 3  # scattered slots, ~3.2 MB each
        slots = torch.randperm(pool - 1)[:tokens] + 1
        if not (slots == 1).any():
            slots[0] = 1  # always in use: the shift hands it to AITER as index 0
        idx = torch.zeros(seqs, win, dtype=torch.int32)  # padded rows: NULL_BLOCK_ID
        idx[:real] = slots.view(real, win).to(torch.int32)
        nacc = torch.ones(seqs, dtype=torch.int32)
        nacc[:real] = torch.tensor(acc, dtype=torch.int32)[torch.randperm(real) % len(acc)]
        if null_slots:  # NULL write slot in seq 0 (reads slot 4); NULL read slot in seq 1
            nacc[0] = win
            idx[0, 2] = 0
            idx[1, nacc[1] - 1] = 0
        cu = torch.full((seqs + 1,), tokens, dtype=torch.int32)
        cu[: real + 1] = torch.arange(real + 1, dtype=torch.int32) * win
        pages = torch.zeros(pool, page, dtype=torch.int8, device=dev)
        conv, ssm = views(pages)
        conv.normal_()
        ssm.normal_(std=0.1)
        rows = n + tail
        defined = torch.zeros(rows, dtype=torch.bool)  # rows stock writes from computed values
        for i in range(real):
            defined[i * win:(i + 1) * win] = bool(idx[i, nacc[i] - 1] > 0)
        defined[n:] = True  # tail: zero in both
        md = GDNAttentionMetadata(
            num_prefills=0, num_prefill_tokens=0, num_decodes=0, num_decode_tokens=0,
            num_spec_decodes=seqs, num_spec_decode_tokens=tokens, num_actual_tokens=n,
            spec_query_start_loc=cu.to(dev), spec_state_indices_tensor=idx.to(dev),
            spec_sequence_masks=torch.ones(seqs, dtype=torch.bool, device=dev),
            spec_token_indx=torch.arange(n, dtype=torch.int32, device=dev),
            non_spec_token_indx=torch.empty(0, dtype=torch.int32, device=dev),
            num_accepted_tokens=nacc.to(dev))
        return dict(md=md, pages=pages, defined=defined.to(dev),
                    qkvz=torch.randn(rows, qkv_dim + hv * dv, device=dev, dtype=bf16),
                    ba=torch.randn(rows, 2 * hv, device=dev, dtype=bf16))

    def stock(st, qkvz, ba, z, out):
        Q.QwenGatedDeltaNetAttention._forward_core_rocm(layer, qkvz, ba, z, out)

    def new(st, qkvz, ba, z, out):
        if not forward_spec(layer, qkvz, ba, out, st["md"]):
            raise RuntimeError("forward_spec declined an all-spec batch")

    def bufs(st):
        rows = st["qkvz"].shape[0]
        return (st["pages"].clone(), st["qkvz"].clone(), st["ba"].clone(),
                torch.empty(rows, hv, dv, device=dev, dtype=bf16),
                torch.full((rows, hv, dv), float("nan"), device=dev, dtype=bf16))

    def call(fn, st, pages, qkvz, ba, z, out):
        layer.kv_cache = views(pages)
        fc = ForwardContext(no_compile_layers={}, attn_metadata={"gdn": st["md"]}, slot_mapping={})
        with override_forward_context(fc):
            fn(st, qkvz, ba, z, out)

    def run(fn, st):
        b = bufs(st)
        call(fn, st, *b)
        return b

    def compare(st, ref, got):
        (rp, rq, _, rz, ro), (gp, gq, _, _, go) = ref, got
        d = st["defined"]
        exact = (torch.equal(rp, gp) and torch.equal(ro[d], go[d])
                 and torch.equal(rz, gq[:, qkv_dim:].view_as(rz)))
        ds = (views(rp)[1] - views(gp)[1]).abs().max().item()
        do = (ro[d].float() - go[d].float()).abs().max().item()
        ok = exact or (torch.allclose(views(rp)[1], views(gp)[1], rtol=1e-5, atol=1e-6)
                       and torch.equal(views(rp)[0], views(gp)[0])
                       and torch.equal(rp[:, cbytes + sbytes:], gp[:, cbytes + sbytes:])
                       and torch.allclose(ro[d].float(), go[d].float(), rtol=1e-2, atol=1e-3))
        ok = ok and torch.equal(rq[:, qkv_dim:], gq[:, qkv_dim:]) and torch.equal(
            rz, gq[:, qkv_dim:].view_as(rz)) and bool((go[st["md"].num_actual_tokens:] == 0).all())
        verdict = ("MATCH (bitwise)" if exact else "MATCH") if ok else "MISMATCH"
        return ok, f"{verdict} max abs diff out {do:.2e} state {ds:.2e}"

    def graph(fn, st):
        b = bufs(st)
        call(fn, st, *b)  # JIT outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call(fn, st, *b)
        return g, b

    def graphed_us(fn, st):
        g, keep = graph(fn, st)  # keep: the captured buffers must outlive the replays
        return _time_us(g.replay)

    failed = False
    for name, real, pad, tail, acc, nulls in (
            ("c1 acc1", 1, 0, 0, [1], False),
            ("c1 acc5", 1, 0, 0, [5], False),
            ("c8 graph-padded to 16", 8, 8, 0, [1, 2, 3, 4, 5], False),
            ("c32 eager tail 16 + NULL slots", 32, 0, 16, [1, 2, 3, 4, 5], True)):
        st = case(real, pad, tail, acc, nulls)
        ok, msg = compare(st, run(stock, st), run(new, st))
        failed |= not ok
        t_stock, t_new = graphed_us(stock, st), graphed_us(new, st)
        print(f"[suffix gdn-mtp] {name} ({real}+{pad} seqs x {win} tokens): {msg} | "
              f"graphed stock {t_stock:.1f} us -> new {t_new:.1f} us", flush=True)

    # Graph safety: capture new on the padded c8 layout, then replay with 5 real
    # sequences and new indices, accepted counts, inputs and states.
    st = case(8, 8, 0, [1, 2, 3, 4, 5])
    g, (pages, qkvz, ba, z, out) = graph(new, st)
    st2 = case(5, 11, 0, [1, 2, 3, 4, 5])
    for k in ("spec_query_start_loc", "spec_state_indices_tensor", "num_accepted_tokens"):
        getattr(st["md"], k).copy_(getattr(st2["md"], k))
    for dst, src in ((pages, st2["pages"]), (qkvz, st2["qkvz"]), (ba, st2["ba"])):
        dst.copy_(src)
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    ok, msg = compare(st2, run(stock, st2), (pages, qkvz, ba, z, out))
    failed |= not ok
    print(f"[suffix gdn-mtp] graph replay (captured 8+8, replayed 5+11, new slots/acc/inputs): "
          f"{msg}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
