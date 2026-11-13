"""Oracle and bench of the GLM RecoverSSM kernels on the GLM-5.3-Flash per-rank KDA shapes (K3 / K4): the stock vLLM fused_recurrent_kda snapshots are the reference.

  python -m suffix_hybrid.glm_recoverssm oracle [--bf16] [--upstream]     bit-exactness (exit 1 on any mismatch), then the timing grid
  python -m suffix_hybrid.glm_recoverssm bench                            34 layers per step: stock against verify + ONE commit launch (graph replays)

oracle = (1) cycles of verify -> 'sampling' (random acceptance) -> commit against the stock snapshot protocol: outputs of every cycle and the committed state bit for bit (several
layers, variable window lengths, null rows); (2) the align cases with a small block size (B = 16, window 8): n0 around the block boundary x acceptance below / at / above the boundary,
the final state must land in the block of the last accepted token (the OLD block when the window crosses and too few tokens are accepted), the boundary state in the block it belongs
to only when the accepted tokens reach it, and NO other block of the pool may change. `--upstream` runs the same with upstream's kernels (tolerance 1e-5 relative instead of equality)."""
from __future__ import annotations

import json
import sys
import types

import torch

H, K, V = 16, 128, 128
LB = -5.0
D = 3 * H * K
CONV_WIDTH_EXTRA = 3        # kernel size 4 -> 3 history columns


def _stock():
    from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
    return fused_recurrent_kda


def _impl(upstream: bool):
    if upstream:
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import KDARecoverSSMCommitContext, KDARecoverSSMRecords, kda_recoverssm_verify
        return (lambda mr, q: KDARecoverSSMRecords.allocate(mr, H, q, K, torch.bfloat16, torch.device("cuda")), kda_recoverssm_verify, KDARecoverSSMCommitContext)
    from suffix_hybrid.glm_recoverssm_kernels import GlmCommitContext, GlmRecords, kda_glm_verify
    return (lambda mr, q: GlmRecords.allocate(mr, H, q, K, torch.device("cuda")), kda_glm_verify, GlmCommitContext)


def tokens(total, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(total, 3 * H * K, device="cuda", dtype=torch.bfloat16, generator=gen)
    q, k, v = qkv.split(H * K, dim=-1)
    return dict(q=q.reshape(1, -1, H, K), k=k.reshape(1, -1, H, K), v=v.reshape(1, -1, H, V),
                g=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16, generator=gen),
                beta=torch.randn(1, total, H, device="cuda", dtype=torch.bfloat16, generator=gen))


def stock_snapshots(tok, state0, lens, a_log, dt_bias):
    """The stock spec kernel from state0 [n, H, V, K] over windows of the given lengths: (outputs, snapshots [n, Tmax, H, V, K])."""
    n, tmax = state0.shape[0], max(lens)
    pool = torch.zeros(1 + n * tmax, H, V, K, device="cuda", dtype=state0.dtype)
    idx = (1 + torch.arange(n * tmax, device="cuda", dtype=torch.int32)).view(n, tmax)
    pool[idx[:, 0].long()] = state0
    cu = torch.zeros(n + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens, device="cuda", dtype=torch.int32), 0)
    out = _stock()(q=tok["q"], k=tok["k"], v=tok["v"], g=tok["g"], beta=tok["beta"], initial_state=pool, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, ssm_state_indices=idx,
                   num_accepted_tokens=torch.ones(n, device="cuda", dtype=torch.int32), sigmoid_beta=True, a_log=a_log, g_bias=dt_bias, compute_gate=True, lower_bound=LB)[0]
    return out, pool[idx.long()], cu


def _eq(a, b, upstream):
    if not upstream:
        return torch.equal(a, b)
    return bool(((a.float() - b.float()).abs().max() <= 1e-5 * max(float(a.float().abs().max()), 1e-6)).item())


def make_ctx(states, convs, a_logs, dt_biases, recs, t, max_rows, ctx_cls):
    adapters = [types.SimpleNamespace(kv_cache=(c, s), recoverssm_records=r, A_log=a.view(H), dt_bias=d.view(H, K), gate_lower_bound=LB, local_num_heads=H, head_dim=K)
                for c, s, r, a, d in zip(convs, states, recs, a_logs, dt_biases)]
    return ctx_cls.create(adapters, spec_query_len=t, max_num_reqs=max_rows, backend="triton")


def cycles(n, lens, n_layers=3, n_cycles=16, null_rows=(), seed=0, state_dtype=torch.float32, upstream=False):
    alloc, verify, ctx_cls = _impl(upstream)
    t = max(lens)
    width = CONV_WIDTH_EXTRA + t - 1
    gens = [torch.Generator(device="cuda").manual_seed(seed + 31 * i) for i in range(n_layers)]
    s = 1 + n * t
    states = [(torch.randn(s, H, V, K, device="cuda", generator=g) * 0.05).to(state_dtype) for g in gens]
    convs = [torch.randn(s, width, D, device="cuda", dtype=torch.bfloat16, generator=g) for g in gens]
    a_logs = [torch.randn(1, 1, H, 1, device="cuda", generator=g) * 0.1 for g in gens]
    dts = [torch.randn(H * K, device="cuda", generator=g) * 0.1 for g in gens]
    recs = [alloc(n, t) for _ in range(n_layers)]
    ctx = make_ctx(states, convs, a_logs, dts, recs, t, n, ctx_cls)
    slot = (1 + torch.arange(n, device="cuda", dtype=torch.int32))                      # the checkpoint slot of each row
    for r in null_rows:
        slot[r] = 0
    cu = torch.zeros(n + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(lens, device="cuda", dtype=torch.int32), 0)
    lens_t = torch.tensor(lens, device="cuda", dtype=torch.int32)
    live = torch.tensor([r not in null_rows for r in range(n)], device="cuda")
    ok = True
    for cyc in range(n_cycles):
        tok = tokens(int(cu[-1]), 1000 + cyc)
        refs = []
        for i in range(n_layers):
            o_ref, snaps, _ = stock_snapshots(tok, states[i][slot.long()].clone(), lens, a_logs[i], dts[i])
            out = verify(tok["q"], tok["k"], tok["v"], tok["g"], tok["beta"], a_logs[i].reshape(-1), dts[i], LB, states[i], recs[i], cu, slot, t) if upstream else \
                verify(tok["q"], tok["k"], tok["v"], tok["g"], tok["beta"], a_logs[i].reshape(-1), dts[i], LB, states[i], recs[i], cu, slot, t)
            rows = torch.cat([torch.arange(int(cu[r]), int(cu[r + 1]), device="cuda") for r in range(n) if r not in null_rows])
            ok &= _eq(o_ref.reshape(-1, H, V)[rows], out.reshape(-1, H, V)[rows], upstream)
            refs.append(snaps)
        acc = torch.minimum((torch.rand(n, device="cuda") * lens_t.float()).long().to(torch.int32) + 1, lens_t)
        ctx.commit(acc, slot, cu)
        for i in range(n_layers):
            want = refs[i][torch.arange(n, device="cuda"), acc.long() - 1][live]
            got = states[i][slot.long()][live]
            ok &= _eq(want, got, upstream)
    return ok


def align_cases(n_layers=2, B=16, t=8, seed=0, state_dtype=torch.float32, upstream=False):
    """Block-boundary states of the commit against the stock snapshots (see the module doc). Returns (ok, number of cases)."""
    alloc, verify, ctx_cls = _impl(upstream)
    cases = [(n0, a) for n0 in (0, 3, B - 9, B - 8, B - 7, B - 5, B - 1, B, B + 3, 2 * B - 2) for a in range(1, t + 1)]
    n, nb = len(cases), 8
    width = CONV_WIDTH_EXTRA + t - 1
    gens = [torch.Generator(device="cuda").manual_seed(seed + 31 * i) for i in range(n_layers)]
    nblocks = 1 + n * nb
    bt = (1 + torch.arange(n * nb, device="cuda", dtype=torch.int32)).view(n, nb)
    n0 = torch.tensor([c[0] for c in cases], device="cuda", dtype=torch.int32)
    acc = torch.tensor([c[1] for c in cases], device="cuda", dtype=torch.int32)
    p1 = ((n0 + t - 1) // B).long()                                                     # column of the running block the verify reads
    src = bt[torch.arange(n, device="cuda"), p1].contiguous()
    cu = torch.arange(0, (n + 1) * t, t, device="cuda", dtype=torch.int32)[: n + 1]
    ok = True
    states = [(torch.randn(nblocks, H, V, K, device="cuda", generator=g) * 0.05).to(state_dtype) for g in gens]
    convs = [torch.randn(nblocks, width, D, device="cuda", dtype=torch.bfloat16, generator=g) for g in gens]
    a_logs = [torch.randn(1, 1, H, 1, device="cuda", generator=g) * 0.1 for g in gens]
    dts = [torch.randn(H * K, device="cuda", generator=g) * 0.1 for g in gens]
    recs = [alloc(n, t) for _ in range(n_layers)]
    ctx = make_ctx(states, convs, a_logs, dts, recs, t, n, ctx_cls)
    tok = tokens(n * t, 4242)
    before = [s_.clone() for s_ in states]
    refs = []
    for i in range(n_layers):
        _, snaps, _ = stock_snapshots(tok, states[i][src.long()].clone(), [t] * n, a_logs[i], dts[i])
        verify(tok["q"], tok["k"], tok["v"], tok["g"], tok["beta"], a_logs[i].reshape(-1), dts[i], LB, states[i], recs[i], cu, src, t)
        refs.append(snaps)
    ctx.commit(acc, src, cu, block_table=bt, num_computed_tokens=n0, mamba_block_size=B)
    for i in range(n_layers):
        expect = before[i].clone()
        for r, (c0, a) in enumerate(cases):
            fin_col = min(max(c0 + a - 1, 0) // B, nb - 1)
            expect[int(bt[r, fin_col])] = refs[i][r, a - 1]
            nxt = (c0 // B + 1) * B
            if c0 + a >= nxt:
                expect[int(bt[r, nxt // B - 1])] = refs[i][r, nxt - c0 - 1]
        ok &= _eq(expect, states[i], upstream)
    return ok, n


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


def timing(n, t, n_layers=34, upstream=False):
    alloc, verify, ctx_cls = _impl(upstream)
    width = CONV_WIDTH_EXTRA + t - 1
    gens = [torch.Generator(device="cuda").manual_seed(5 + i) for i in range(n_layers)]
    s = 1 + n * t
    states = [(torch.randn(s, H, V, K, device="cuda", generator=g) * 0.05) for g in gens]
    stock_states = [x.clone() for x in states]
    convs = [torch.randn(s, width, D, device="cuda", dtype=torch.bfloat16, generator=g) for g in gens]
    a_logs = [torch.randn(1, 1, H, 1, device="cuda", generator=g) * 0.1 for g in gens]
    dts = [torch.randn(H * K, device="cuda", generator=g) * 0.1 for g in gens]
    recs = [alloc(n, t) for _ in range(n_layers)]
    ctx = make_ctx(states, convs, a_logs, dts, recs, t, n, ctx_cls)
    toks = [tokens(n * t, 9 + i) for i in range(n_layers)]
    idx = (1 + torch.arange(n * t, device="cuda", dtype=torch.int32)).view(n, t)
    slot = (1 + torch.arange(n, device="cuda", dtype=torch.int32))
    cu = torch.arange(0, (n + 1) * t, t, device="cuda", dtype=torch.int32)[: n + 1]
    acc = torch.randint(1, t + 1, (n,), device="cuda", dtype=torch.int32)

    def f_stock():
        for i in range(n_layers):
            _stock()(q=toks[i]["q"], k=toks[i]["k"], v=toks[i]["v"], g=toks[i]["g"], beta=toks[i]["beta"], initial_state=stock_states[i], use_qk_l2norm_in_kernel=True,
                     cu_seqlens=cu, ssm_state_indices=idx, num_accepted_tokens=acc, sigmoid_beta=True, a_log=a_logs[i], g_bias=dts[i], compute_gate=True, lower_bound=LB)

    def f_verify():
        for i in range(n_layers):
            verify(toks[i]["q"], toks[i]["k"], toks[i]["v"], toks[i]["g"], toks[i]["beta"], a_logs[i].reshape(-1), dts[i], LB, states[i], recs[i], cu, slot, t)

    def f_commit():
        ctx.commit(acc, slot, cu)

    a, b, c = bench_graph(f_stock) / 1000, bench_graph(f_verify) / 1000, bench_graph(f_commit) / 1000
    r = {"kernels": "upstream" if upstream else "exact", "N": n, "T": t, "layers": n_layers, "stock_ms": round(a, 3), "verify_ms": round(b, 3), "commit_ms": round(c, 3),
         "total_ms": round(b + c, 3), "saved_ms": round(a - b - c, 3)}
    return r


def main(argv) -> int:
    if not argv or argv[0] not in ("oracle", "bench"):
        print(__doc__)
        return 2
    upstream = "--upstream" in argv
    bad = 0
    if argv[0] == "oracle":
        dtypes = [("fp32", torch.float32)] + ([("bf16", torch.bfloat16)] if "--bf16" in argv else [])
        for name, dt in dtypes:
            for n, lens, nulls in ((1, [4], ()), (1, [8], ()), (8, [8] * 8, (2,)), (32, [8] * 32, (2,)), (6, [8, 5, 1, 8, 3, 8], (2,)), (9, [8, 2, 8, 8, 4, 8, 1, 8, 8], (2,))):
                ok = cycles(n, lens, null_rows=nulls, state_dtype=dt, upstream=upstream)
                bad += not ok
                print(json.dumps({"check": "cycles", "state": name, "n": n, "lens": lens if n < 10 else f"{lens[0]}x{n}", "null_rows": list(nulls), "bit_exact" if not upstream else "close": ok}), flush=True)
            ok, cnt = align_cases(state_dtype=dt, upstream=upstream)
            bad += not ok
            print(json.dumps({"check": "align boundary cases", "state": name, "cases": cnt, "bit_exact" if not upstream else "close": ok}), flush=True)
    for n, t in ((1, 4), (1, 8), (8, 4), (8, 8), (16, 8), (32, 4), (32, 8)):
        print(json.dumps(timing(n, t, upstream=upstream)), flush=True)
    print("ORACLE", "PASS" if not bad else f"FAIL ({bad} checks)", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
