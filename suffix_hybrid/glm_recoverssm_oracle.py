"""Oracle and bench of the upstream RecoverSSM kernels on the GLM-5.3-Flash per-rank KDA shapes (K3 / K4): the stock vLLM fused_recurrent_kda is the reference.

  python -m suffix_hybrid.glm_recoverssm oracle      numerics over 24 cycles with random acceptance (state and outputs against the stock snapshots), then the timing grid
  python -m suffix_hybrid.glm_recoverssm bench       34 layers per step: stock against verify + one commit launch (graph replays)

The kernels are upstream's (vllm.models.kimi_k3.nvidia.ops.recoverssm); they are equal to the stock snapshots up to fp32 rounding, not bit-identical (rsqrt, sigmoid form, reverse accumulation).
Exit 1 when a state differs by more than 1e-5 relative or an output by more than 2 bf16 ulps of its range."""

import json, sys, types
import torch
from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext, KDARecoverSSMRecords, kda_recoverssm_verify

H, K, V = 16, 128, 128
LB = -5.0
CONV_DIM, CONV_LEN = 6144, 10


def make(n, t, n_layers, seed=0):
    layers = []
    for l in range(n_layers):
        gen = torch.Generator(device="cuda").manual_seed(seed + 31 * l)
        s = n * t + 2
        state = torch.randn(s, H, V, K, device="cuda", dtype=torch.float32, generator=gen) * 0.05
        a_log = torch.randn(1, 1, H, 1, device="cuda", dtype=torch.float32, generator=gen) * 0.1
        dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32, generator=gen) * 0.1
        conv = torch.randn(s, CONV_LEN, CONV_DIM, device="cuda", dtype=torch.bfloat16, generator=gen)   # SD layout like the layer (the commit transposes it)
        rec = KDARecoverSSMRecords.allocate(n, H, t, K, torch.bfloat16, torch.device("cuda"))
        layers.append(dict(state=state, k3_state=state.clone(), a_log=a_log, dt_bias=dt_bias, conv=conv, rec=rec))
    idx = (1 + torch.arange(n * t, device="cuda", dtype=torch.int32)).view(n, t)
    cu = torch.arange(0, (n + 1) * t, t, device="cuda", dtype=torch.int32)[: n + 1]
    adapters = [types.SimpleNamespace(kv_cache=(L["conv"], L["k3_state"]), recoverssm_records=L["rec"], A_log=L["a_log"].view(H), dt_bias=L["dt_bias"].view(H, K),
                                      gate_lower_bound=LB, local_num_heads=H, head_dim=K) for L in layers]
    ctx = KDARecoverSSMCommitContext.create(adapters, spec_query_len=t, max_num_reqs=n)
    return layers, idx, cu, ctx


def tokens(n, t, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(n * t, 3 * H * K, device="cuda", dtype=torch.bfloat16, generator=gen)
    q, k, v = qkv.split(H * K, dim=-1)
    return dict(q=q.reshape(1, -1, H, K), k=k.reshape(1, -1, H, K), v=v.reshape(1, -1, H, V),
                g=torch.randn(1, n * t, H, K, device="cuda", dtype=torch.bfloat16, generator=gen),
                beta=torch.randn(1, n * t, H, device="cuda", dtype=torch.bfloat16, generator=gen))


def stock(L, tok, idx, cu, acc):
    return fused_recurrent_kda(q=tok["q"], k=tok["k"], v=tok["v"], g=tok["g"], beta=tok["beta"], initial_state=L["state"], use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                               ssm_state_indices=idx, num_accepted_tokens=acc, sigmoid_beta=True, a_log=L["a_log"], g_bias=L["dt_bias"], compute_gate=True, lower_bound=LB)[0]


def k3_verify(L, tok, idx, cu, t):
    return kda_recoverssm_verify(tok["q"], tok["k"], tok["v"], tok["g"], tok["beta"], L["a_log"].view(H), L["dt_bias"], LB, L["k3_state"], L["rec"], cu, idx[:, 0].contiguous(), t)


def numerics(n, t, n_layers=3, cycles=24):
    layers, idx, cu, ctx = make(n, t, n_layers)
    ar = torch.arange(n, device="cuda")
    acc = torch.ones(n, device="cuda", dtype=torch.int32)
    worst_o = worst_s = 0.0
    rel_o = rel_s = 0.0
    for c in range(cycles):
        tok = tokens(n, t, 100 + c)
        for L in layers:
            o_s = stock(L, tok, idx, cu, acc)
            o_k = k3_verify(L, tok, idx, cu, t)
            d = (o_s.float() - o_k.float()).abs()
            worst_o = max(worst_o, float(d.max())); rel_o = max(rel_o, float(d.max() / o_s.float().abs().max()))
        acc = torch.randint(1, t + 1, (n,), device="cuda", dtype=torch.int32)
        ctx.commit(acc, idx[:, 0].contiguous(), cu)
        for L in layers:
            ref = L["state"][idx.long()[ar, acc.long() - 1]]
            got = L["k3_state"][idx[:, 0].long()]
            d = (ref - got).abs()
            worst_s = max(worst_s, float(d.max())); rel_s = max(rel_s, float(d.max() / ref.abs().max()))
        # next cycle: stock reads column acc-1 of ITS pool, k3 reads slot 0 (committed); keep both pools aligned for the next cycle's comparison
    return {"n": n, "t": t, "cycles": cycles, "max_abs_out": worst_o, "max_rel_out": rel_o, "max_abs_state": worst_s, "max_rel_state": rel_s}


def bench(fn, reps=3):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    best = 1e9
    for _ in range(9):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); e.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / reps)
    return best


def timing(n, t, n_layers=34):
    layers, idx, cu, ctx = make(n, t, n_layers)
    toks = [tokens(n, t, 7 + i) for i in range(n_layers)]
    acc = torch.randint(1, t + 1, (n,), device="cuda", dtype=torch.int32)
    s0 = idx[:, 0].contiguous()

    def f_stock():
        for i, L in enumerate(layers):
            stock(L, toks[i], idx, cu, acc)

    def f_verify():
        for i, L in enumerate(layers):
            k3_verify(L, toks[i], idx, cu, t)

    def f_commit():
        ctx.commit(acc, s0, cu)

    a, b, c = bench(f_stock) / 1000, bench(f_verify) / 1000, bench(f_commit) / 1000
    return {"n": n, "t": t, "layers": n_layers, "stock_ms": round(a, 3), "k3_verify_ms": round(b, 3), "k3_commit_ms": round(c, 3), "k3_total_ms": round(b + c, 3), "saved_ms": round(a - b - c, 3)}




def main(argv) -> int:
    if not argv or argv[0] not in ("oracle", "bench"):
        print(__doc__)
        return 2
    bad = 0
    if argv[0] == "oracle":
        for n, t in ((1, 8), (8, 8), (32, 8), (32, 4)):
            r = numerics(n, t)
            bad += not (r["max_rel_state"] < 1e-5 and r["max_rel_out"] < 2 ** -6)
            print(json.dumps(r), flush=True)
    for n, t in ((1, 4), (1, 8), (8, 4), (8, 8), (16, 8), (32, 4), (32, 8)):
        print(json.dumps(timing(n, t)), flush=True)
    print("ORACLE", "PASS" if not bad else f"FAIL ({bad} cases)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
