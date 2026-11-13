"""Oracle and bench of the lazy-state KDA verify (K3 / K4): the stock vLLM fused_recurrent_kda is the exact reference.

  python -m suffix_hybrid.kda_lazy oracle [--graph] [--bf16]      bit-exactness over many cycles (exit 1 on any mismatch)
  python -m suffix_hybrid.kda_lazy bench                          34 layers per step, stock against lazy, N x T grid

The exactness protocol runs the stock kernel (snapshot per token, initial state from column acc-1) and the lazy kernel side by side over several cycles with random
acceptance and compares every output and the committed state at every window boundary bit for bit: several layers, variable window lengths, null rows (block 0),
rows that sit out a cycle (keep their acceptance), a prefill clear of a block that holds records, fp32 and bf16 state. `--graph` captures one lazy step of all
layers in a CUDA graph and replays it over the cycles (the decode path of a full graph)."""
from __future__ import annotations

import json
import sys

import torch

H, K, V = 16, 128, 128
LB = -5.0


def _stock():
    from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
    return fused_recurrent_kda


def make_layer(n, lens, seed, state_dtype=torch.float32):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tmax = max(lens)
    s = n * tmax + 2
    state = (torch.randn(s, H, V, K, device="cuda", dtype=torch.float32, generator=gen) * 0.05).to(state_dtype)
    a_log = torch.randn(1, 1, H, 1, device="cuda", dtype=torch.float32, generator=gen) * 0.1
    dt_bias = torch.randn(H * K, device="cuda", dtype=torch.float32, generator=gen) * 0.1
    idx = (1 + torch.arange(n * tmax, device="cuda", dtype=torch.int32)).view(n, tmax)
    cu = torch.zeros(n + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens, device="cuda", dtype=torch.int32), 0)
    return dict(state=state, a_log=a_log, g_bias=dt_bias, idx=idx, cu=cu, lens=lens)


def new_tokens(cu, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tot = int(cu[-1])
    qkv = torch.randn(tot, 3 * H * K, device="cuda", dtype=torch.bfloat16, generator=gen)   # q|k|v columns of one projection, like the layer
    q, k, v = qkv.split(H * K, dim=-1)
    return dict(q=q.reshape(1, -1, H, K), k=k.reshape(1, -1, H, K), v=v.reshape(1, -1, H, V),
                g=torch.randn(1, tot, H, K, device="cuda", dtype=torch.bfloat16, generator=gen),
                beta=torch.randn(1, tot, H, device="cuda", dtype=torch.bfloat16, generator=gen))


def stock_call(layer, tok, state, idx, cu, acc):
    return _stock()(q=tok["q"], k=tok["k"], v=tok["v"], g=tok["g"], beta=tok["beta"], initial_state=state, use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                    ssm_state_indices=idx, num_accepted_tokens=acc, sigmoid_beta=True, a_log=layer["a_log"], g_bias=layer["g_bias"], compute_gate=True,
                    lower_bound=LB)[0]


def lazy_call(lz, layer, tok, idx, cu, acc, out=None, replay_only=False):
    return lz.run(tok["q"], tok["k"], tok["v"], tok["g"], tok["beta"], cu, idx, acc, layer["a_log"], layer["g_bias"], LB, out=out, replay_only=replay_only)


def exactness(n, lens, n_layers=3, cycles=12, null_rows=(), skip_rows=(), clear_at=None, seed=0, state_dtype=torch.float32):
    """See the module doc. Returns (outputs_exact, committed_state_exact, worst_abs_diff)."""
    from suffix_hybrid.kda_lazy_kernels import LazyKDA
    layers = [make_layer(n, lens, seed + 17 * i, state_dtype) for i in range(n_layers)]
    for L in layers:
        for r in null_rows:
            L["idx"][r, :] = 0
    stock_state = [L["state"].clone() for L in layers]
    lazy = [LazyKDA(L["state"].clone(), spec_len=max(lens), max_rows=max(n, 64)) for L in layers]
    tmax = max(lens)
    lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
    ar = torch.arange(n, device="cuda")
    acc = torch.ones(n, device="cuda", dtype=torch.int32)        # first step after a prefill: committed state = column 0, nothing to replay
    live = torch.tensor([r not in null_rows for r in range(n)], device="cuda")
    ok_o = ok_s = True
    worst = 0.0
    for cyc in range(cycles):
        toks = [new_tokens(L["cu"], 1000 * (cyc + 1) + i) for i, L in enumerate(layers)]
        sched = [r for r in range(n) if not (cyc % 3 == 1 and r in skip_rows)]
        if clear_at is not None and cyc == clear_at[0]:
            # a new request takes the block of row clear_at[1]: its prefill writes the final state to column 0 and clears the header
            for i, L in enumerate(layers):
                blk = L["idx"][clear_at[1], 0].long()
                fresh = (torch.randn(H, V, K, device="cuda", generator=torch.Generator(device="cuda").manual_seed(77 + i)) * 0.05).to(state_dtype)
                stock_state[i][blk] = fresh
                lazy[i].state[blk] = fresh
                lazy[i].clear(blk.view(1))
            acc[clear_at[1]] = 1
        for i, L in enumerate(layers):
            sub_idx = L["idx"][sched]
            sub_lens = [lens[r] for r in sched]
            sub_cu = torch.zeros(len(sched) + 1, device="cuda", dtype=torch.int32)
            sub_cu[1:] = torch.cumsum(torch.tensor(sub_lens, device="cuda", dtype=torch.int32), 0)
            sel = torch.cat([torch.arange(int(L["cu"][r]), int(L["cu"][r]) + lens[r], device="cuda") for r in sched])
            tk = {key: val[:, sel] for key, val in toks[i].items()}
            sub_acc = acc[sched].contiguous()
            o_stock = stock_call(L, tk, stock_state[i], sub_idx, sub_cu, sub_acc)
            o_lazy = lazy_call(lazy[i], L, tk, sub_idx, sub_cu, sub_acc)
            keep = [j for j, r in enumerate(sched) if r not in null_rows]
            rows = torch.cat([torch.arange(int(sub_cu[j]), int(sub_cu[j + 1]), device="cuda") for j in keep])
            a, b = o_stock.reshape(-1, H, V)[rows], o_lazy.reshape(-1, H, V)[rows]
            eq = torch.equal(a, b)
            ok_o &= eq
            if not eq:
                worst = max(worst, float((a.float() - b.float()).abs().max()))
        new_acc = (torch.rand(n, device="cuda") * lens_t.float()).long().clamp(min=0, max=tmax - 1).to(torch.int32) + 1
        new_acc = torch.minimum(new_acc, lens_t)
        for r in sched:
            acc[r] = new_acc[r]
        # boundary: the lazy committed state (replay-only launch on a clone) against the stock snapshot of the last accepted token
        for i, L in enumerate(layers):
            tmp = LazyKDA(lazy[i].state.clone(), spec_len=max(lens), max_rows=max(n, 64))
            tmp.hdr.copy_(lazy[i].hdr)
            lazy_call(tmp, L, toks[i], L["idx"], L["cu"], acc, replay_only=True)
            idx = L["idx"].long()
            stock_c = stock_state[i][idx[ar, acc.long() - 1]][live]
            lazy_c = tmp.state[idx[:, 0]][live]
            same = torch.equal(stock_c, lazy_c)
            ok_s &= same
            if not same:
                worst = max(worst, float((stock_c.float() - lazy_c.float()).abs().max()))
    return ok_o, ok_s, worst


def exactness_graph(n, t, n_layers=3, cycles=10, seed=0):
    """One lazy step of all layers captured in a CUDA graph (static inputs, like the decode path of a full graph), replayed over the cycles: outputs and committed
    states against the stock kernel run eagerly on the same inputs."""
    from suffix_hybrid.kda_lazy_kernels import LazyKDA
    layers = [make_layer(n, [t] * n, seed + 17 * i) for i in range(n_layers)]
    stock_state = [L["state"].clone() for L in layers]
    lazy = [LazyKDA(L["state"].clone(), spec_len=t, max_rows=max(n, 64)) for L in layers]
    static = [new_tokens(L["cu"], 5 + i) for i, L in enumerate(layers)]
    static_acc = torch.ones(n, device="cuda", dtype=torch.int32)
    static_out = [torch.empty(1, n * t, H, V, device="cuda", dtype=torch.bfloat16) for _ in layers]
    warm = [LazyKDA(L["state"].clone(), spec_len=t, max_rows=max(n, 64)) for L in layers]      # compile before the capture, on throwaway state
    for i, L in enumerate(layers):
        lazy_call(warm[i], L, static[i], L["idx"], L["cu"], static_acc, out=static_out[i])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i, L in enumerate(layers):
            lazy_call(lazy[i], L, static[i], L["idx"], L["cu"], static_acc, out=static_out[i])
    ok_o = ok_s = True
    worst = 0.0
    ar = torch.arange(n, device="cuda")
    for cyc in range(cycles):
        fresh = [new_tokens(L["cu"], 900 + 10 * cyc + i) for i, L in enumerate(layers)]
        for i in range(n_layers):
            for key in static[i]:
                static[i][key].copy_(fresh[i][key])
        o_stock = [stock_call(L, fresh[i], stock_state[i], L["idx"], L["cu"], static_acc) for i, L in enumerate(layers)]
        g.replay()
        torch.cuda.synchronize()
        for i in range(n_layers):
            eq = torch.equal(o_stock[i].reshape(-1, H, V), static_out[i].reshape(-1, H, V))
            ok_o &= eq
            if not eq:
                worst = max(worst, float((o_stock[i].float() - static_out[i].float()).abs().max()))
        acc = torch.randint(1, t + 1, (n,), device="cuda", dtype=torch.int32)
        for i, L in enumerate(layers):
            tmp = LazyKDA(lazy[i].state.clone(), spec_len=t, max_rows=max(n, 64))
            tmp.hdr.copy_(lazy[i].hdr)
            lazy_call(tmp, L, fresh[i], L["idx"], L["cu"], acc, replay_only=True)
            idx = L["idx"].long()
            same = torch.equal(stock_state[i][idx[ar, acc.long() - 1]], tmp.state[idx[:, 0]])
            ok_s &= same
        static_acc.copy_(acc)
    return ok_o, ok_s, worst


def bench_graph(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(9):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e) * 1000 / reps)
    return best


def timing(n, t, n_layers=34):
    from suffix_hybrid.kda_lazy_kernels import LazyKDA
    layers = [make_layer(n, [t] * n, 5 + i) for i in range(n_layers)]
    toks = [new_tokens(L["cu"], 9 + i) for i, L in enumerate(layers)]
    stock_state = [L["state"] for L in layers]
    lazy = [LazyKDA(L["state"].clone(), spec_len=t, max_rows=max(n, 64)) for L in layers]
    acc = torch.randint(1, t + 1, (n,), device="cuda", dtype=torch.int32)
    for z in lazy:
        z.hdr.fill_(t & 0xFF)          # an old window of t tokens in buffer 0

    def stock_all():
        for i, L in enumerate(layers):
            stock_call(L, toks[i], stock_state[i], L["idx"], L["cu"], acc)

    def lazy_all():
        for i, L in enumerate(layers):
            lazy_call(lazy[i], L, toks[i], L["idx"], L["cu"], acc)

    r = {"N": n, "T": t, "layers": n_layers, "stock_ms": round(bench_graph(stock_all) / 1000, 3), "lazy_ms": round(bench_graph(lazy_all) / 1000, 3)}
    r["speedup"] = round(r["stock_ms"] / r["lazy_ms"], 2)
    r["saved_ms"] = round(r["stock_ms"] - r["lazy_ms"], 3)
    return r


CASES = [dict(n=1, lens=[4]), dict(n=1, lens=[8]), dict(n=8, lens=[8] * 8, null_rows=(2,)), dict(n=32, lens=[8] * 32, null_rows=(2,)),
         dict(n=32, lens=[4] * 32, null_rows=(5,)), dict(n=6, lens=[8, 5, 1, 8, 3, 8], null_rows=(2,)),
         dict(n=9, lens=[8, 2, 8, 8, 4, 8, 1, 8, 8], null_rows=(2,), skip_rows=(1, 4, 6)), dict(n=8, lens=[8] * 8, skip_rows=(0, 3), clear_at=(5, 3))]


def main(argv) -> int:
    if not argv or argv[0] not in ("oracle", "bench"):
        print(__doc__)
        return 2
    if argv[0] == "bench":
        for n, t in ((1, 4), (1, 8), (8, 4), (8, 8), (16, 8), (32, 4), (32, 8)):
            print(json.dumps(timing(n, t)), flush=True)
        return 0
    bad = 0
    dtypes = [("fp32", torch.float32)] + ([("bf16", torch.bfloat16)] if "--bf16" in argv else [])
    for name, dt in dtypes:
        for c in CASES:
            ok_o, ok_s, worst = exactness(state_dtype=dt, **c)
            bad += not (ok_o and ok_s)
            print(json.dumps({"state": name, "n": c["n"], "lens": c["lens"] if c["n"] < 10 else f"{c['lens'][0]}x{c['n']}", "null_rows": list(c.get("null_rows", ())),
                              "skip_rows": list(c.get("skip_rows", ())), "clear_at": c.get("clear_at"), "outputs_bit_exact": ok_o,
                              "committed_state_bit_exact": ok_s, "max_abs_diff_if_not": worst}), flush=True)
    if "--graph" in argv:
        for n, t in ((1, 8), (8, 8), (32, 8), (32, 4)):
            ok_o, ok_s, worst = exactness_graph(n, t)
            bad += not (ok_o and ok_s)
            print(json.dumps({"graph": True, "n": n, "t": t, "outputs_bit_exact": ok_o, "committed_state_bit_exact": ok_s, "max_abs_diff_if_not": worst}), flush=True)
    print("ORACLE", "PASS" if not bad else f"FAIL ({bad} cases)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
