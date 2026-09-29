# SPDX-License-Identifier: Apache-2.0
"""CPU proofs for suffix_hybrid/mtp_tune (online idle-time MTP tuning).

(a) zero-init adapter is a bitwise no-op (layer forward + MTP replay)
(b) frozen-expert / frozen-linear backward == autograd on dequantized weights
(c) language-shifted traffic: trainer raises replayed acceptance and the
    gate promotes; on already-fit traffic it does not promote
(d) capture: label join, rejected rows, prompt ctx, gaps, departure,
    eviction, dropped steps, served-draft parity rows
(e) engine idle hook yields to arriving work (<= one micro-step)
(f) in-place promotion visible to a previously captured closure
(g) TP=2 (gloo, 2 processes): sharded adapters + autograd collectives give
    the TP=1 loss, replay and gradients
"""
import os
import queue
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import mtp_toy as T  # noqa: E402

from suffix_hybrid.mtp_tune import Config  # noqa: E402
from suffix_hybrid.mtp_tune import lora as L  # noqa: E402
from suffix_hybrid.mtp_tune.capture import Store, wrap_propose  # noqa: E402
from suffix_hybrid.mtp_tune.gate import Gate, lang_tag  # noqa: E402
from suffix_hybrid.mtp_tune.scheduler import EngineIdle, Tuner  # noqa: E402
from suffix_hybrid.mtp_tune.train import (  # noqa: E402
    FrozenExperts, FrozenLinear, WindowAttention, build_functional, chain_replay,
    depth1_loss, expert_weights, experts_handle)

K = 3
TARGETS = ("*fc_embedding", "*fc_hidden", "*self_attn.qkv_proj", "*self_attn.o_proj",
           "*shared_expert.gate_up_proj", "*shared_expert.down_proj")


# ---------------------------------------------------------------------------
# synthetic serving traffic through the real propose wrapper
# ---------------------------------------------------------------------------
def native_propose(input_batch, attn_metadata, slot_mappings, last_hidden_states,
                   aux_hidden_states, num_sampled, num_rejected, last_sampled,
                   next_prefill_tokens, temperature, seeds, dp_sync=None, dummy_run=False,
                   skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False):
    n = input_batch.num_reqs
    return (torch.arange(n * K).view(n, K) + 7) % T.V   # deterministic "drafts"


def lang_seq(lang, n, seed, active=12, p_rule=0.95):
    """Tokens of language A (ids 0..) or B (ids 48..): x_{t+1} = perm(x_t)
    with prob p_rule, else uniform over the language's active tokens."""
    base = 0 if lang == "A" else 48
    perm = list(range(active))
    random.Random(lang).shuffle(perm)
    rng = random.Random(seed)
    x = [rng.randrange(active)]
    for _ in range(n - 1):
        x.append(perm[x[-1]] if rng.random() < p_rule else rng.randrange(active))
    return [base + t for t in x]


def drive(prop, seqs, plen=8, seed=0, skip=None, table=None):
    """Serve `seqs` ({rid: tokens}) step by step: one prefill chunk, then
    decode steps of K+1 query rows with random rejections. Garbage tokens /
    hidden in rejected rows. `skip` = {(rid, step)} steps where rid is left
    out of the batch while its position still advances (a capture gap)."""
    rng = random.Random(seed)
    table = T.hidden_table() if table is None else table
    pos = {r: 0 for r in seqs}
    step = 0
    while pos:
        ids, qs, comp, plens, toks, poss, rej = [], [0], [], [], [], [], []
        for r in list(pos):
            s, p = seqs[r], pos[r]
            if p == 0:
                q, nrej = plen, 0
            else:
                q, nrej = K + 1, rng.randrange(K + 1)
            if p + q > len(s):
                del pos[r]
                continue
            valid = q - nrej
            pos[r] = p + valid
            if skip and (r, step) in skip:
                continue
            ids.append(r)
            comp.append(p)
            plens.append(plen)
            toks += s[p:p + valid] + [T.V - 1] * nrej
            poss += list(range(p, p + q))
            rej.append(nrej)
            qs.append(qs[-1] + q)
        step += 1
        if not ids:
            continue
        tok = torch.tensor(toks)
        ib = SimpleNamespace(req_ids=ids, num_reqs=len(ids), query_start_loc_np=np.array(qs),
                             num_computed_tokens_np=np.array(comp),
                             prefill_len_np=np.array(plens), input_ids=tok.int(),
                             positions=torch.tensor(poss))
        hid = table[tok] + 0.01 * torch.randn(len(toks), table.shape[1])
        n = len(ids)
        prop(ib, None, None, hid, None, torch.ones(n, dtype=torch.long), torch.tensor(rej),
             None, None, None, None)
    prop(SimpleNamespace(req_ids=[], num_reqs=0, query_start_loc_np=np.array([0]),
                         num_computed_tokens_np=np.array([]), prefill_len_np=np.array([]),
                         input_ids=torch.zeros(0, dtype=torch.int32),
                         positions=torch.zeros(0, dtype=torch.long)),
         None, None, torch.zeros(0, T.HC * T.H), None, torch.zeros(0), torch.zeros(0, dtype=torch.long),
         None, None, None, None)


# ---------------------------------------------------------------------------
# (a) zero-init no-op
# ---------------------------------------------------------------------------
def test_zero_init_adapter_is_bitwise_noop():
    m = T.build(fp8_dense=True)
    x = torch.randn(5, T.H)
    lin = m.model.fc_embedding            # FP8-block quantized dense
    before = lin(x)
    lo = L.attach(lin, "fc_embedding", rank=16)
    assert torch.equal(lin(x), before)
    assert lo.B.abs().sum() == 0 and lo.A.abs().sum() > 0
    plain = torch.nn.Linear(T.H, 7, bias=False)    # no quant_method: forward wrap
    y = plain(x)
    L.attach(plain, "plain", rank=4)
    assert torch.equal(plain(x), y)

    fam_ref = build_functional(T.build(seed=3))
    m2 = T.build(seed=3)
    L.attach_all(m2, TARGETS, rank=8)
    fam = build_functional(m2)
    h = T.hidden_table()[torch.randint(0, T.V, (24,))]
    tok, pos = torch.randint(0, T.V, (24,)), torch.arange(24) % 12 + 100
    a = fam_ref.step(h, tok, pos, WindowAttention(2, 12))
    b = fam.step(h, tok, pos, WindowAttention(2, 12))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# (b) frozen backward == autograd on dequantized weights
# ---------------------------------------------------------------------------
def test_frozen_experts_backward_matches_autograd():
    m = T.build()
    handle = dict(experts_handle(m.model.layers[0].mlp.experts), act_qdq=False)
    assert handle["kind"] == "fp8"
    torch.manual_seed(0)
    x = torch.randn(9, T.H, requires_grad=True)
    tw = torch.rand(9, T.TOPK, requires_grad=True)
    ids = torch.randint(0, T.E, (9, T.TOPK))
    ids[0, 1] = -1                                   # off-rank pair contributes 0
    dy = torch.randn(9, T.H)
    y = FrozenExperts.apply(x, tw, ids, handle)
    (y * dy).sum().backward()

    xr = x.detach().clone().requires_grad_(True)
    twr = tw.detach().clone().requires_grad_(True)
    ref = torch.zeros(9, T.H)
    for t in range(9):
        for k in range(T.TOPK):
            e = int(ids[t, k])
            if e < 0:
                continue
            w13, w2 = expert_weights(handle, e)
            g, u = (w13 @ xr[t]).chunk(2)
            ref[t] = ref[t] + twr[t, k] * (w2 @ (torch.nn.functional.silu(g) * u))
    (ref * dy).sum().backward()
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(tw.grad, twr.grad, rtol=1e-4, atol=1e-5)
    assert tw.grad[0, 1] == 0


def test_frozen_linear_probe_backward_on_fp8_layer():
    lin = T.build(fp8_dense=True).model.fc_embedding
    x = torch.randn(6, T.H, requires_grad=True)
    dy = torch.randn(6, T.H)
    (FrozenLinear.apply(x, lin) * dy).sum().backward()
    w = lin.weight.float().reshape(1, 128, 1, 128) * lin.weight_scale_inv[:, None, :, None]
    xr = x.detach().clone().requires_grad_(True)
    (torch.nn.functional.linear(xr, w.reshape(T.H, T.H)) * dy).sum().backward()
    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# (c) end to end: capture -> idle micro-steps -> gate
# ---------------------------------------------------------------------------
def _cfg(**kw):
    base = dict(rate=1.0, window=24, ctx=4, mem_gib=0.05, heldout_pct=30, rank=16,
                targets=TARGETS, lr=3e-3, accum=1, max_windows=4, eval_every=60,
                eval_windows=48, min_eval_windows=4, min_train_windows=4, margin=0.05,
                min_anchors=150, log_s=1e9, lang=False, rollback_file="",
                step_ms=1e9)   # micro-batch = max_windows: timing-independent
    base.update(kw)
    return Config(**base)


def _traffic(tuner, lang, n_req=40, length=90, seed=0):
    prop = wrap_propose(native_propose, tuner.store)
    drive(prop, {f"{lang}{seed}-{i}": lang_seq(lang, length, seed * 1000 + i)
                 for i in range(n_req)}, seed=seed)


def _run_until_eval(tuner, max_steps=2000):
    evals = tuner.counts["eval"]
    done = None
    for _ in range(max_steps):
        r = tuner.step()
        assert not r.get("disabled")
        if tuner.counts["eval"] > evals and not tuner.eval_queue:
            done = tuner.gate.last
            break
    assert done is not None, "no gate evaluation happened"
    return done


def _merge_into_base(tuner):
    """Bake the live adapters into the toy's float base weights (the
    'checkpoint head trained on language A')."""
    for name, lo in tuner.loras.items():
        lin = tuner.fam.m.get_submodule(name)
        with torch.no_grad():
            lin.weight += (lo.A.float() @ lo.B.float() * lo.scale).T.to(lin.weight.dtype)
        lo.zero_()


def test_tuner_promotes_on_shifted_language_not_on_fit_one():
    torch.manual_seed(0)
    m = T.build(seed=5)
    tuner = Tuner(m, K, T.HC * T.H, _cfg(), dtype=torch.float32)
    # 1) "pretraining": the head learns language A, promoted, merged into base
    _traffic(tuner, "A", seed=1)
    first = _run_until_eval(tuner)
    assert first["promote"] and first["cand"] > first["live"] + 0.5, first
    for _ in range(8):                     # train on until the gate stops promoting
        if not _run_until_eval(tuner)["promote"]:
            break
    else:
        pytest.fail("pretraining never converged")
    _merge_into_base(tuner)

    # 2) already-fit traffic (A): fresh candidate cannot beat live -> no promotion
    tuner = Tuner(m, K, T.HC * T.H, _cfg(), dtype=torch.float32)
    _traffic(tuner, "A", seed=2)
    fit = _run_until_eval(tuner)
    assert fit["live"] > 1.5, fit             # base is good on A
    assert not fit["promote"], fit
    assert tuner.gate.promotions == 0

    # 3) language-shifted traffic (B): base is poor, candidate learns, promoted
    tuner = Tuner(m, K, T.HC * T.H, _cfg(), dtype=torch.float32)
    _traffic(tuner, "B", seed=3)
    shifted = _run_until_eval(tuner)
    assert shifted["live"] < 0.5, shifted
    assert shifted["promote"] and shifted["cand"] > shifted["live"] + 0.5, shifted
    assert tuner.gate.promotions == 1
    # live adapters now carry the candidate: replay with live == candidate result
    b = tuner._batch("eval", tuner.store.alive_windows(heldout=True)[:4])
    live = {n: (lo.A, lo.B) for n, lo in tuner.loras.items()}
    r1 = chain_replay(tuner.fam, b, live, K)
    r2 = chain_replay(tuner.fam, b, tuner.trainer.serving_params(), K)
    assert torch.equal(r1["correct"], r2["correct"])
    # parity: served drafts were recorded on anchor rows
    assert (b["d1"] >= 0).any()
    assert "promotions=1" in tuner.line()


def test_tuner_rollback_and_shadow(tmp_path):
    flag = tmp_path / "rb"
    m = T.build(seed=5)
    tuner = Tuner(m, K, T.HC * T.H, _cfg(rollback_file=str(flag), shadow=True), dtype=torch.float32)
    _traffic(tuner, "B", seed=4)
    res = _run_until_eval(tuner)
    assert res["cand"] > res["live"] + 0.5 and tuner.gate.promotions == 0   # shadow
    for lo in tuner.loras.values():
        lo.B.fill_(0.5)
    flag.write_text("")
    tuner.step()
    assert tuner.rolled_back and all(lo.B.abs().sum() == 0 for lo in tuner.loras.values())


def test_trainer_exception_disables_without_raising(monkeypatch):
    tuner = Tuner(T.build(), K, T.HC * T.H, _cfg(), dtype=torch.float32)
    monkeypatch.setenv("SUFFIX_MTP_TUNE_FAULT_RANK", "0")
    assert tuner.step() == {"worked": False, "disabled": True}
    assert tuner.step()["disabled"]


# ---------------------------------------------------------------------------
# (d) capture: label join, eviction, drops
# ---------------------------------------------------------------------------
def _store(**kw):
    base = dict(width=T.HC * T.H, k=K, cap_rows=4096, window=16, rate=1.0, ctx=4,
                min_rows=8, heldout_pct=0, dtype=torch.float32)
    base.update(kw)
    return Store(**base)


def test_capture_windows_join_labels_and_drop_rejected_rows():
    st = _store()
    seq = lang_seq("A", 120, 0)
    drive(wrap_propose(native_propose, st), {"r": seq}, plen=20)
    ws = [st.materialize(w) for w in st.alive_windows(False)]
    assert len(ws) >= 4
    L_ = 16 + K + 1
    first = ws[0]
    assert first["pos"][0] == 16                 # plen 20 - ctx 4: older prompt rows skipped
    for w in ws:
        p0 = int(w["pos"][0])
        assert torch.equal(w["pos"], torch.arange(p0, p0 + len(w["pos"])))
        assert w["tok"].tolist() == seq[p0:p0 + len(w["tok"])]     # rejected rows gone
        torch.testing.assert_close(w["h"], T.hidden_table()[w["tok"].long()], atol=0.1, rtol=0)
        assert torch.equal(w["anchor"], w["pos"] >= 19)
    full = [w for w in ws if len(w["tok"]) == L_]
    assert all(int(b["pos"][0]) == int(a["pos"][0]) + 16 for a, b in zip(full, full[1:]))
    assert len(ws[-1]["tok"]) >= 8               # departure emitted the tail
    d1 = torch.cat([w["d1"] for w in ws])
    assert (d1 >= 0).sum() > 10


def test_capture_gap_restarts_window_and_eviction_is_fifo():
    st = _store()
    seq = lang_seq("A", 200, 1)
    drive(wrap_propose(native_propose, st), {"r": seq}, plen=8, skip={("r", 12)})
    assert st.stats["gaps"] == 1
    for w in st.alive_windows(False):
        mw = st.materialize(w)
        p0 = int(mw["pos"][0])
        assert mw["tok"].tolist() == seq[p0:p0 + len(mw["tok"])]

    small = _store(cap_rows=64)
    drive(wrap_propose(native_propose, small), {f"r{i}": lang_seq("B", 80, i) for i in range(6)})
    rec = list(small.train)
    alive = small.alive_windows(False)
    assert small.stats["evicted_windows"] > 0 or len(alive) < small.stats["windows_train"]
    for w in alive:
        assert small.materialize(w) is not None
    assert all(small.arena.alive(int(w["rows"][0])) == (small.materialize(w) is not None) for w in rec)


class _NotDone:
    def query(self):
        return False


def test_capture_drops_step_when_slots_busy_never_waits():
    st = _store()
    prop = wrap_propose(native_propose, st)
    ib = SimpleNamespace(req_ids=["a"], num_reqs=1, query_start_loc_np=np.array([0, 4]),
                         num_computed_tokens_np=np.array([10]), prefill_len_np=np.array([4]),
                         input_ids=torch.arange(4).int(), positions=torch.arange(10, 14))
    args = (None, None, torch.zeros(4, T.HC * T.H), None, torch.ones(1), torch.zeros(1, dtype=torch.long),
            None, None, None, None)
    for _ in range(2):
        prop(ib, *args)
        st.inflight[-1]["ev"] = _NotDone()   # side-stream copy still in flight
    out = prop(ib, *args)
    assert out.shape == (1, K) and st.stats["dropped_steps"] == 1
    assert len(st.inflight) == 2
    st.inflight[0]["ev"] = None
    st.absorb()
    assert len(st.inflight) == 1        # FIFO: stops at the first busy slot


def test_capture_failure_never_breaks_propose():
    st = _store()
    prop = wrap_propose(lambda *a, **kw: "native", st)
    bad = SimpleNamespace(req_ids=["x"], num_reqs=1)          # missing fields -> capture raises
    args = (None, None, torch.zeros(1, 4), None, None, None, None, None, None, None)
    assert prop(bad, *args) == "native"
    assert prop(bad, *args) == "native"                        # capture now off, still serving


# ---------------------------------------------------------------------------
# (e) engine idle hook
# ---------------------------------------------------------------------------
class FakeCore:
    def __init__(self):
        self.input_queue, self.work, self.handled = queue.Queue(), False, []

    def has_work(self):
        return self.work

    def is_running(self):
        return True

    def _handle_client_request(self, *req):
        self.handled.append(req)
        self.work = True


def test_engine_idle_hook_yields_to_arriving_work():
    core = FakeCore()
    calls = []

    def rpc(stats):
        calls.append(stats)
        if len(calls) == 3:          # a request arrives mid micro-step
            core.input_queue.put(("ADD", "req"))
            time.sleep(0.02)
        return {"worked": True, "ms": 1.0}

    idle = EngineIdle(_cfg(idle_grace_ms=1), rpc=rpc)
    idle.run(core)
    assert len(calls) == 3                       # stopped right after that micro-step
    assert 20 <= idle.max_added_ms < 1000
    assert not core.input_queue.empty()          # left for the engine's own loop

    core2 = FakeCore()
    core2.input_queue.put(("ADD", "x"))
    idle2 = EngineIdle(_cfg(idle_grace_ms=1), rpc=lambda s: pytest.fail("trained while busy"))
    idle2.run(core2)                             # queue non-empty: no micro-step
    core2.input_queue.get()
    core2.work = True
    idle2.run(core2)                             # has_work: no micro-step

    core3 = FakeCore()
    idle3 = EngineIdle(_cfg(idle_grace_ms=1), rpc=lambda s: {"worked": False, "disabled": True})
    idle3.run(core3)
    assert not idle3.on
    idle4 = EngineIdle(_cfg(idle_grace_ms=1), rpc=lambda s: {"worked": False})
    idle4.run(FakeCore())                        # nothing to train: back to blocking
    assert idle4.on and idle4.rpcs == 1


# ---------------------------------------------------------------------------
# (f) in-place contract
# ---------------------------------------------------------------------------
def test_inplace_promotion_visible_to_captured_closure():
    lin = T.build().model.fc_hidden
    lo = L.attach(lin, "fc_hidden", rank=4)
    x = torch.randn(3, T.H)
    captured = lin.quant_method.apply     # "graph": holds the adapter tensors by reference
    a_ptr, b_ptr = lo.A.data_ptr(), lo.B.data_ptr()
    base = captured(lin, x)
    new = {"fc_hidden": (torch.randn_like(lo.A), torch.randn_like(lo.B))}
    Gate(K).promote({"fc_hidden": lo}, new)
    assert (lo.A.data_ptr(), lo.B.data_ptr()) == (a_ptr, b_ptr)
    expect = base + (x @ new["fc_hidden"][0]) @ new["fc_hidden"][1] * lo.scale
    torch.testing.assert_close(captured(lin, x), expect)
    # a rebind would NOT reach a captured graph (address-bound): demonstrate
    old_b = lo.B
    frozen = lambda: (x @ lo.A) @ old_b   # noqa: E731 - graph captured old address
    lo.B = torch.zeros_like(lo.B)
    assert frozen().abs().sum() > 0


def test_lang_tag():
    assert lang_tag("Das ist nicht die Antwort, aber wir können es noch einmal versuchen.") == "de"
    assert lang_tag("This is not the answer, but we can try it once more with feeling.") == "other"
    assert lang_tag("ok") == "?"


# ---------------------------------------------------------------------------
# (g) TP=2 over gloo
# ---------------------------------------------------------------------------
class GlooGroup:
    def __init__(self):
        import torch.distributed as dist
        self.d = dist
        self.world_size, self.rank_in_group = dist.get_world_size(), dist.get_rank()

    def all_reduce(self, x):
        y = x.clone()
        self.d.all_reduce(y)
        return y

    def all_gather(self, x, dim=-1):
        parts = [torch.empty_like(x) for _ in range(self.world_size)]
        self.d.all_gather(parts, x.contiguous())
        return torch.cat(parts, dim)

    def broadcast_tensor_dict(self, d=None, src=0):
        obj = [d]
        self.d.broadcast_object_list(obj, src)
        return obj[0]


def _tp_batch():
    g = torch.Generator().manual_seed(11)
    b, t = 2, 14
    tok = torch.randint(0, T.V, (b, t), generator=g)
    return dict(h=T.hidden_table()[tok] + 0.1 * torch.randn(b, t, T.HC * T.H, generator=g),
                tok=tok, pos=torch.arange(t).repeat(b, 1) + 50,
                anchor=torch.ones(b, t, dtype=torch.bool), lens=torch.tensor([t, t - 3]))


def _tp_run(rank, world, group):
    """Loss, replay and global-indexed adapter grads on this rank."""
    m = T.build(rank, world, group, seed=9)
    loras = L.attach_all(m, TARGETS, rank=4, tp_rank=rank)
    fam = build_functional(m, group)
    params = {}
    for name, lo in loras.items():
        lin = m.get_submodule(name)
        g = torch.Generator().manual_seed(len(name))
        kf = lo.A.shape[0] * (world if lo.kind == "row" else 1)
        nf = lo.B.shape[1] * (world if lo.kind == "col" else 1)
        a, b = torch.randn(kf, 4, generator=g) * 0.1, torch.randn(4, nf, generator=g) * 0.1
        if hasattr(lin, "gather_output"):        # column layer: B columns sharded
            b = b[:, lin.shard_idx]
        if hasattr(lin, "input_is_parallel"):    # row layer: A rows sharded
            a = a[lin.shard_idx]
        params[name] = (a.requires_grad_(True), b.requires_grad_(True))
    batch = _tp_batch()
    loss, _ = depth1_loss(fam, batch, params)
    loss.backward()
    L.allreduce_grads([(lo, params[n][0 if lo.replicated_factor() == "A" else 1])
                       for n, lo in loras.items() if lo.replicated_factor()], fam.g)
    with torch.no_grad():
        rep = chain_replay(fam, batch, params, K)
    grads = {}
    for name, lo in loras.items():
        lin = m.get_submodule(name)
        ga, gb = params[name][0].grad, params[name][1].grad
        grads[name] = dict(kind=lo.kind, idx=getattr(lin, "shard_idx", None), A=ga, B=gb)
    return dict(loss=float(loss.detach()), correct=rep["correct"], pred1=rep["pred1"], grads=grads)


def _tp_child(rank, world, path):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{path}/pg", rank=rank, world_size=world)
    try:
        torch.save(_tp_run(rank, world, GlooGroup()), f"{path}/r{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_tp2_matches_tp1(tmp_path, monkeypatch):
    import torch.multiprocessing as mp
    for key in [k for k in os.environ if k.startswith("SUFFIX_")]:
        monkeypatch.delenv(key)      # spawned children run sitecustomize
    ref = _tp_run(0, 1, None)
    mp.spawn(_tp_child, args=(2, str(tmp_path)), nprocs=2, join=True)
    ranks = [torch.load(tmp_path / f"r{r}.pt", weights_only=False) for r in range(2)]
    for r in ranks:
        assert abs(r["loss"] - ref["loss"]) < 1e-4 * max(1.0, abs(ref["loss"]))
        assert torch.equal(r["pred1"], ref["pred1"]) and torch.equal(r["correct"], ref["correct"])
    for name, g in ref["grads"].items():
        parts = [r["grads"][name] for r in ranks]
        kind = parts[0]["kind"]
        for fac in "AB":
            full = g[fac]
            sharded = (kind == "col" and fac == "B") or (kind == "row" and fac == "A")
            for p in parts:
                if sharded:
                    got = p[fac]
                    want = full[:, p["idx"]] if fac == "B" else full[p["idx"]]
                else:
                    got, want = p[fac], full          # replicated: all-reduced, identical
                torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-6,
                                           msg=f"{name}.{fac} ({kind})")
        if kind in ("col", "row"):
            assert torch.equal(parts[0]["A" if kind == "col" else "B"],
                               parts[1]["A" if kind == "col" else "B"])
