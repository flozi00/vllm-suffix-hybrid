# SPDX-License-Identifier: Apache-2.0
"""Oracle for SUFFIX_ROCM_QK_FUSED: vLLM's fused split + QK GemmaRMSNorm + partial NeoX
mRoPE + gate copy (vllm/model_executor/layers/fused_qk_norm_rope.py, one Triton launch) on
the Qwen3.8-Flash-Next QSA layers vs the eager path that ROCm runs today (the AMD
Qwen4ExpQSAAttention enables the fused kernel only on CUDA and only text-only; inductor
compiles the eager path into ~4-5 launches per layer).

Shapes: 24 q heads (+ 24 gate), 2 kv heads, head_dim 256, rotary_dim 64, interleaved
mRoPE sections (11, 11, 10), theta 1e7, positions (3, T) with image-like rows where
T / H / W differ. Prints bitwise share and max |diff| in units of bf16 ulp of the eager
result, graph replay == eager, graphed us/call fused vs eager.

    python -m suffix_hybrid.kernels.qk_fused_rocm   # boot gate qk_fused_bench
"""
import torch

MARK = "[suffix qk-fused]"
H, KV, D = 24, 2, 256
ROPE = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
        "rope_theta": 10000000, "rope_type": "default"}


def _graph_us(fn, iters: int = 50):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / iters, out


def main() -> int:
    from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope

    dev = "cuda"
    torch.set_default_device(dev)
    torch.manual_seed(0)
    rope = get_rope(head_size=D, max_position=262144, rope_parameters=ROPE, dtype=torch.bfloat16)
    q_norm, k_norm = GemmaRMSNorm(D, eps=1e-6), GemmaRMSNorm(D, eps=1e-6)
    with torch.no_grad():
        q_norm.weight.normal_(0, 0.3)
        k_norm.weight.normal_(0, 0.3)
    q_norm, k_norm = q_norm.to(torch.bfloat16), k_norm.to(torch.bfloat16)
    print(f"{MARK} rope {type(rope).__name__} rotary_dim {rope.rotary_dim} "
          f"interleaved {getattr(rope, 'mrope_interleaved', None)} dtype {rope.cos_sin_cache.dtype}",
          flush=True)

    def eager(qkv, pos):  # Qwen3NextAttention._project_qkv_gate's non-fused branch
        q_gate, k, v = qkv.split([H * D * 2, KV * D, KV * D], dim=-1)
        q, gate = torch.chunk(q_gate.view(-1, H, 2 * D), 2, dim=-1)
        q, gate = q.reshape(-1, H * D), gate.reshape(-1, H * D)
        q = q_norm(q.view(-1, H, D)).view(-1, H * D)
        k = k_norm(k.view(-1, KV, D)).view(-1, KV * D)
        q, k = rope(pos, q, k)
        return q, k, gate

    def fused(qkv, pos):
        q_gate, k, _ = qkv.split([H * D * 2, KV * D, KV * D], dim=-1)
        return fused_qk_rmsnorm_rope_gate(
            q_gate, k, q_norm.weight, k_norm.weight, rope.cos_sin_cache, pos, 1e-6, H, KV, D,
            rope.rotary_dim, mrope_section=rope.mrope_section, norm_beta=1.0)

    failed = False
    for m in (1, 5, 32, 40, 160, 1024):
        qkv = torch.randn(m, (2 * H + 2 * KV) * D, dtype=torch.bfloat16)
        base = torch.randint(0, 200000, (m,))
        pos = torch.stack([base, base, base])  # text rows: T = H = W
        img = torch.arange(m) % 3 == 0  # image-like rows: distinct H / W
        pos[1, img] = base[img] // 7
        pos[2, img] = base[img] % 1013
        ref = [t.clone() for t in eager(qkv, pos)]
        out = [t.clone() for t in fused(qkv, pos)]
        parts = []
        for name, r, o in zip(("q", "k", "gate"), ref, out):
            ulp = (r.float().abs() * 2.0 ** -7).clamp_min(2.0 ** -133)
            parts.append(f"{name} bitwise {100 * (r == o).float().mean().item():.2f}% "
                         f"max {((r.float() - o.float()).abs() / ulp).max().item():.1f} ulp")
            failed |= ((r.float() - o.float()).abs() / ulp).max().item() > 2.0
        fus_us, g_out = _graph_us(lambda: fused(qkv, pos))
        replay_ok = all(torch.equal(a, b) for a, b in zip(g_out, out))
        failed |= not replay_ok
        eag_us, _ = _graph_us(lambda: eager(qkv, pos))
        print(f"{MARK} M={m}: {' | '.join(parts)} | graph == eager {replay_ok} | graphed us: "
              f"eager {eag_us:.1f} -> fused {fus_us:.1f}", flush=True)
    print(f"{MARK} {'PASS' if not failed else 'FAIL'} (<= 2 bf16 ulp vs eager)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
