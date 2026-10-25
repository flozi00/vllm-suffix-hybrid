"""CPU checks of the DeepSeek-V4.1 SFT twin (suffix_hybrid/sft/dsv41.py) on a tiny fake
module tree with the vLLM attribute surface: shapes, every path engaged, gradient reach,
the clamped expert backward and the window/compressed/sink attention against a dense reference."""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from suffix_hybrid.mtp_tune.lora import SingleGroup
from suffix_hybrid.mtp_tune.train import FrozenExperts
from suffix_hybrid.sft import dsv41 as D

torch.manual_seed(0)
H, HD, QL, HC, E, I = 64, 128, 32, 4, 8, 16


class Rope(nn.Module):
    def __init__(self, rot=64):
        super().__init__()
        self.rotary_dim = rot
        inv = 1.0 / (10000 ** (torch.arange(0, rot, 2).float() / rot))
        f = torch.arange(4096).float()[:, None] * inv[None]
        self.cos_sin_cache = torch.cat([f.cos(), f.sin()], -1)

    def forward_native(self, pos, q, k=None, inverse=False):
        cs = self.cos_sin_cache[pos]
        cos, sin = cs.chunk(2, -1)
        cos = cos.repeat_interleave(2, -1).unsqueeze(-2)
        sin = sin.repeat_interleave(2, -1).unsqueeze(-2)
        sin = -sin if inverse else sin
        r = q[..., -self.rotary_dim:]
        x1, x2 = r[..., ::2], r[..., 1::2]
        rot = torch.stack([-x2, x1], -1).flatten(-2)
        out = (r * cos + rot * sin).to(q.dtype)
        return torch.cat([q[..., :-self.rotary_dim], out], -1), None


class Lin(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k) / k ** 0.5, requires_grad=False)
        self.A = nn.Parameter(torch.randn(k, 4) * 0.1)
        self.B = nn.Parameter(torch.zeros(4, n))

    def forward(self, x):
        xf = x.float()
        return (F.linear(xf, self.weight) + (xf @ self.A) @ self.B).to(x.dtype)


class Norm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(n), requires_grad=False)
        self.variance_epsilon = 1e-6
        self.rms_norm_eps = 1e-6

    def forward(self, x):
        return D._rms(x, self.weight, 1e-6)


def attn(ratio, compressor, indexer, owns_k=True):
    a = nn.Module()
    a.fused_wqa_wkv, a.wq_b, a.wo_b = Lin(H, QL + HD), Lin(QL, 2 * HD), Lin(64, H)
    a.wo_a = nn.Module()
    a.wo_a.weight = torch.randn(64, 2 * HD) / HD ** 0.5      # 1 group: 2 heads x HD -> 64
    a.q_norm, a.kv_norm = Norm(QL), Norm(HD)
    a.q_lora_rank, a.head_dim, a.n_local_heads, a.n_local_groups, a.o_lora_rank = QL, HD, 2, 1, 64
    a.eps, a.scale, a.window_size, a.compress_ratio = 1e-6, HD ** -0.5, 8, ratio
    a.attn_sink = torch.randn(2)
    a.rotary_emb = a.indexer_rotary_emb = Rope()
    a.compressor = a.indexer = None
    if compressor:
        c = nn.Module()
        c.fused_wkv_wgate = Lin(H, HD * ratio)
        c.norm, c.rms_norm_eps = Norm(HD), 1e-6
        a.compressor = c
    if indexer:
        ix = nn.Module()
        ix.topk_tokens, ix.n_head, ix.head_dim, ix.owns_k = 4, 2, 64, owns_k
        ix.wq_b, ix.weights_proj, ix.wk, ix.k_norm = Lin(QL, 128), Lin(H, 2), Lin(HD, 64), Norm(64)
        a.indexer = ix
    return a


def layer(i, ratio, comp, idx, eng=False):
    L = nn.Module()
    L.hc_mult, L.rms_norm_eps, L.hc_eps, L.hc_post_alpha, L.hc_sinkhorn_iters = HC, 1e-6, 1e-6, 2.0, 5
    for p in ("attn", "ffn"):
        setattr(L, f"hc_{p}_fn", torch.randn(24, HC * H) * 0.02)
        setattr(L, f"hc_{p}_scale", torch.rand(3))
        setattr(L, f"hc_{p}_base", torch.randn(24) * 0.1)
    L.hc_attn_fn_broadcast = L.hc_attn_fn.view(24, HC, H).sum(1)
    L.attn_norm, L.ffn_norm = Norm(H), Norm(H)
    L.attn = attn(ratio, comp, idx)
    f = nn.Module()
    f.gate = nn.Module()
    f.gate.weight, f.gate.e_score_correction_bias, f.gate.tid2eid = torch.randn(E, H) * 0.1, torch.zeros(E), None
    f.n_activated_experts, f.renormalize, f.routed_scaling_factor, f.swiglu_limit = 2, True, 1.5, 0.5
    f.scoring_func = "sqrtsoftplus"
    re_ = nn.Module()
    re_.w13_weight, re_.w2_weight = torch.randn(E, 2 * I, H) * 0.3, torch.randn(E, H, I) * 0.3
    f.experts = nn.Module()
    f.experts.routed_experts = re_
    se = nn.Module()
    se.gate_up_proj, se.down_proj = Lin(H, 2 * I), Lin(I, H)
    f.shared_experts = se
    L.ffn = f
    L.engram = None
    if eng:
        g = nn.Module()
        g.eps, g.clamp_value, g.layer_hash_index = 1e-6, 1e-6, 0
        g.q_weight, g.k_weight = torch.randn(HC, H), torch.randn(HC, H)
        table = torch.randn(50, 3, 8)
        g.embed_tokens = lambda idx: table[idx[:, 0]]
        g.wkv = Lin(24, (HC + 1) * H)
        L.engram = g
    return L


def model():
    inner = nn.Module()
    spec = [(0, 0, 0, 0), (0, 0, 0, 1), (2, 1, 1, 0), (2, 0, 0, 0), (1, 1, 1, 0), (1, 0, 1, 0), (1, 0, 0, 0)]
    inner.layers = nn.ModuleList([layer(i, r, c, x, e) for i, (r, c, x, e) in enumerate(spec)])
    emb = torch.randn(100, H)
    inner.embed_input_ids = lambda ids: emb[ids]
    inner.norm = Norm(H)
    hs = types.SimpleNamespace(lookback_depth=3)
    inner.engram_hash = lambda ids, pos, qsl, dead, lb, lbd, a, b: (ids[:, None, None] * 7 + pos[:, None, None]) % 50
    inner.engram_hash = type("HS", (), {"lookback_depth": 3, "__call__": lambda self, ids, pos, *a: (ids[:, None, None] * 7 + pos[:, None, None]) % 50})()
    return inner


def tm_for(segs):
    t = types.SimpleNamespace(g=SingleGroup(), segments=segs, s=types.SimpleNamespace())
    t._checkpointed = lambda fn, **kw: fn
    return t


def test_forward_and_grad_reach():
    inner = model()
    segs = [(0, 13), (13, 30)]
    ids = torch.randint(3, 100, (30,))
    pos = torch.cat([torch.arange(13), torch.arange(17)])
    h = D.forward(tm_for(segs), inner, ids, pos)
    assert h.shape == (30, H) and torch.isfinite(h).all()
    h.square().sum().backward()
    for n, p in inner.named_parameters():
        if n.endswith(".B") and not any(k in n for k in (".engram.", ".compressor.", ".indexer.")):
            assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_segments_independent():
    """Packed segments never see each other: changing segment 1's tokens leaves segment 2's
    rows unchanged (same packing, so the same reductions: any difference is a leak, not
    rounding; packed-vs-alone comparisons flip fp8 KV / bf16 roundings across platforms)."""
    inner = model()
    ids = torch.randint(3, 100, (30,))
    ids2 = ids.clone()
    ids2[:13] = torch.randint(3, 100, (13,))
    pos = torch.cat([torch.arange(13), torch.arange(17)])
    segs = [(0, 13), (13, 30)]
    with torch.no_grad():
        a = D.forward(tm_for(segs), inner, ids, pos)
        b = D.forward(tm_for(segs), inner, ids2, pos)
    assert not torch.allclose(a[:13], b[:13])
    torch.testing.assert_close(a[13:], b[13:], atol=1e-6, rtol=1e-6)


def test_indexer_long_matches_select_all_when_topk_covers():
    """With topk >= visible entries the indexer path equals the select-all path."""
    inner = model()
    for L in inner.layers:
        if L.attn.indexer is not None:
            L.attn.indexer.topk_tokens = 1000
    ids, pos = torch.randint(3, 100, (24,)), torch.arange(24)
    with torch.no_grad():
        a = D.forward(tm_for([(0, 24)]), inner, ids, pos)
        for L in inner.layers:
            if L.attn.indexer is not None:
                L.attn.indexer.topk_tokens = 4     # forces the scored path (24 entries > 4)
        b = D.forward(tm_for([(0, 24)]), inner, ids, pos)
    assert not torch.allclose(a, b)                 # sparse selection changes the output


def test_attend_dense_reference():
    torch.manual_seed(1)
    t, h, d, W, r = 10, 2, 16, 4, 2
    q, kv, C = torch.randn(t, h, d), torch.randn(t, d), torch.randn(t // r, d)
    sink = torch.randn(h)
    pos = torch.arange(t)
    for sel in (None, torch.tensor([[0, -1]] * t)):
        cm = D._cmask(pos[3:], C.shape[0], r, None if sel is None else sel[3:])
        out = D._attend(q[3:], kv[0:], 0, C, cm, 3, sink, 0.25, W)
        for i, p in enumerate(range(3, t)):
            cj = [j for j in range((p + 1) // r) if sel is None or j == 0]
            keys = [kv[u] for u in range(max(0, p - W + 1), p + 1)] + [C[j] for j in cj]
            K = torch.stack(keys)
            s = torch.cat([(q[p] @ K.T) * 0.25, sink[:, None]], -1)
            ref = torch.softmax(s, -1)[:, :-1] @ K
            torch.testing.assert_close(out[i], ref, atol=1e-5, rtol=1e-5)


def test_clamped_experts_backward():
    torch.manual_seed(2)
    hd = dict(kind="float", w13=torch.randn(3, 8, 6), w2=torch.randn(3, 6, 4), limit=0.7)
    x = torch.randn(5, 6, dtype=torch.float64, requires_grad=True).float().detach().requires_grad_(True)
    tw = torch.rand(5, 2, requires_grad=True)
    ids = torch.randint(0, 3, (5, 2))
    y = FrozenExperts.apply(x, tw, ids, hd)
    gy = torch.randn_like(y)
    gx, gt = torch.autograd.grad(y, (x, tw), gy)
    x2, tw2 = x.detach().requires_grad_(True), tw.detach().requires_grad_(True)
    ref = torch.zeros(5, 6)
    for t in range(5):
        for k in range(2):
            gu = hd["w13"][ids[t, k]] @ x2[t]
            g, u = gu[:4].clamp(max=0.7), gu[4:].clamp(-0.7, 0.7)
            ref = ref.index_add(0, torch.tensor([t]), (tw2[t, k] * (hd["w2"][ids[t, k]] @ (F.silu(g) * u)))[None])
    rx, rt = torch.autograd.grad(ref, (x2, tw2), gy)
    torch.testing.assert_close(gx, rx, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gt, rt, atol=1e-4, rtol=1e-4)


def test_host_ckpt_grads_match_plain():
    """_HostCkpt (inputs/outputs parked on the host, one backward pass) gives the same input and
    leaf gradients as no checkpointing, with an int carry and a None among the outputs."""
    from suffix_hybrid.sft.model import _HostCkpt
    torch.manual_seed(3)
    w = torch.randn(6, 6, requires_grad=True)          # stands in for the LoRA flat param
    x0 = torch.randn(4, 6, requires_grad=True)

    def fn(x, c):
        y = torch.tanh(x @ w) + c
        return y, (y.sum(-1) > 0).long(), None, y * 2

    def run(ckpt):
        w.grad = None
        x = x0 * 1.5
        c = x0[:, :1] * 0.5
        dummy = torch.zeros(1, requires_grad=True)
        out = _HostCkpt.apply(fn, "cpu", dummy, None, x, c) if ckpt else fn(x, c)
        (out[0].square().sum() + out[3].sum()).backward()
        return w.grad.clone(), x0.grad.clone()
    a = run(False)
    x0.grad = None
    b = run(True)
    for p, q in zip(a, b):
        torch.testing.assert_close(p, q)
