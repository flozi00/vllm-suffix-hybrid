# SPDX-License-Identifier: Apache-2.0
"""suffix_hybrid.sft on CPU for the LFM2-MoE layout (vLLM lfm2_moe.py): gated
short-conv layers (ShortConv writes into a caller-owned `output`), GQA
attention with `out_proj`, dense SwiGLU `w13`/`w2` in the first layers, a
sparse block with sigmoid scoring + selection-only expert bias + renormalize +
routed scaling factor and no shared expert.

Same contract as test_sft.py: serving-path ops cut autograd or raise, the
oracle (REF mode) is independent pure torch; TrainMode loss and every adapter
gradient must equal it."""
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import test_sft as T
from suffix_hybrid.sft import model as M

H, D, NH, NKV, I, NE, TOPK, IE, V, VPAD = T.H, T.D, T.NH, T.NKV, T.I, T.NE, T.TOPK, T.IE, T.V, T.VPAD
CK = 3          # conv_L_cache
SCALE = 2.5     # routed_scaling_factor
REF = T.REF


def col(w):
    return T.Lin(w, kind="col", idx=torch.arange(w.shape[0]))


def row(w):
    return T.Lin(w, kind="row", idx=torch.arange(w.shape[1]))


class ShortConv(nn.Module):
    """vLLM ShortConv surface: forward(hidden_states, output) writes `output`."""

    def __init__(self, ws):
        super().__init__()
        self.L_cache, self.bias = CK, False
        self.in_proj = col(ws["in"])
        self.conv = nn.Module()
        self.conv.weight = nn.Parameter(ws["conv"].clone().unsqueeze(1), requires_grad=False)   # [C, 1, K]
        self.conv.bias = None
        self.out_proj = row(ws["out"])

    def get_state_shape(self):          # makes check_supported look at us (must be allowed)
        return ((H, CK - 1),)

    def forward(self, hidden_states, output):
        if not REF["on"]:
            raise RuntimeError("conv state-cache path called in training")
        b, c, x = self.in_proj(hidden_states)[0].chunk(3, -1)
        w = self.conv.weight[:, 0]                                    # [C, K]
        rows = []
        for s, e in REF["segs"]:
            bx = (b * x)[s:e]
            xp = torch.cat([bx.new_zeros(CK - 1, bx.shape[1]), bx])
            rows.append(torch.stack([(xp[t:t + CK] * w.t()).sum(0) for t in range(e - s)]))
        output[:hidden_states.shape[0]] = self.out_proj(c * torch.cat(rows))[0]


class LfmAttn(nn.Module):
    def __init__(self, ws, name):
        super().__init__()
        self.qkv_proj = col(ws["qkv"])
        self.out_proj = row(ws["o"])
        self.rotary_emb = T.Rope()
        self.attn = T.Attention(NH, NKV, name)

    def forward(self, pos, x):
        q, k, v = self.qkv_proj(x)[0].split([NH * D, NKV * D, NKV * D], -1)
        q, k = self.rotary_emb(pos, q, k)
        return self.out_proj(self.attn(q, k, v))[0]


class Mlp(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.w13 = col(ws["gu"])
        self.act_fn = T.SiluAndMul()
        self.w2 = row(ws["down"])

    def forward(self, x):
        return self.w2(self.act_fn(self.w13(x)[0]))[0]


class SigmoidExperts(T.RoutedExperts):
    def __init__(self, w13, w2):
        super().__init__(w13, w2, None)
        self.scoring_func, self.use_grouped_topk, self.num_expert_group = "sigmoid", True, 1
        self.routed_scaling_factor, self.custom_routing_function = 1.0, None


class SparseMoe(nn.Module):
    """Lfm2MoeSparseMoeBlock surface."""

    def __init__(self, ws, g):
        super().__init__()
        self.gate = T.Lin(ws["router"])
        self.gate.e_score_correction_bias = nn.Parameter(0.3 * torch.randn(NE, generator=g),
                                                         requires_grad=False)
        self.experts = nn.Module()
        self.experts.routed_experts = SigmoidExperts(ws["w13"], ws["w2"])
        self.experts.routed_experts.routed_scaling_factor = SCALE
        self.experts.routed_scaling_factor = 1.0      # runner side (apply_routed_scale_to_output=False)

    def forward(self, x):
        if not REF["on"]:
            raise RuntimeError("fused MoE runner called in training")
        re_ = self.experts.routed_experts
        score = torch.sigmoid(self.gate(x)[0].float())
        ids = (score + self.gate.e_score_correction_bias.float()).topk(TOPK, -1).indices
        tw = score.gather(-1, ids)
        tw = tw / tw.sum(-1, keepdim=True) * re_.routed_scaling_factor
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(TOPK):
                e = int(ids[t, j])
                gu = re_.w13_weight[e] @ x[t]
                out[t] = out[t] + tw[t, j] * (re_.w2_weight[e] @ (F.silu(gu[:IE]) * gu[IE:]))
        return out


class Layer(nn.Module):
    def __init__(self, kind, ffn, ws, g, idx):
        super().__init__()
        self.kind = kind
        self.operator_norm, self.ffn_norm = T.Norm(H, g), T.Norm(H, g)
        if kind == "conv":
            self.short_conv = ShortConv(ws)
        else:
            self.self_attn = LfmAttn(ws, f"model.layers.{idx}.self_attn.attn")
        self.feed_forward = Mlp(ws) if ffn == "dense" else SparseMoe(ws["moe"], g)

    def forward(self, positions, hidden_states, residual, **kw):
        if residual is None:
            residual, h = hidden_states, self.operator_norm(hidden_states)
        else:
            h, residual = self.operator_norm(hidden_states, residual)
        if self.kind == "conv":
            out = torch.empty_like(h)
            self.short_conv(h, out)
        else:
            out = self.self_attn(positions, h)
        h, residual = self.ffn_norm(out, residual)
        return self.feed_forward(h), residual


class Inner(nn.Module):
    PLAN = (("conv", "dense"), ("conv", "moe"), ("attn", "moe"))

    def __init__(self, ws, g):
        super().__init__()
        self.do_not_compile = False
        self.embed_tokens = T.Embed(ws["emb"], None, 0)
        self.layers = nn.ModuleList([Layer(k, f, ws["layers"][i], g, i)
                                     for i, (k, f) in enumerate(self.PLAN)])
        self.embedding_norm = T.Norm(H, g)

    def __call__(self, *a, **kw):
        if not (self.do_not_compile or REF["on"]):
            raise RuntimeError("compiled forward called in training")
        return self.forward(*a, **kw)

    def forward(self, input_ids, positions, intermediate_tensors=None):
        h, res = self.embed_tokens(input_ids), None
        for layer in self.layers:
            h, res = layer(positions=positions, hidden_states=h, residual=res)
        return self.embedding_norm(h, res)[0]


class LM(nn.Module):
    def __init__(self, ws, g):
        super().__init__()
        self.model = Inner(ws, g)
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(ws["head"].clone(), requires_grad=False)
        self.lm_head.quant_method = T.UnquantizedEmbeddingMethod()
        self.lm_head.shard_indices = type("S", (), dict(org_vocab_start_index=0, org_vocab_end_index=V))()
        self.logits_processor = type("L", (), dict(scale=1.0, soft_cap=None))()


def weights(seed=11):
    g = torch.Generator().manual_seed(seed)

    def r(*s, sc=0.2):
        return torch.randn(*s, generator=g) * sc
    layers = [dict(**{"in": r(3 * H, H), "conv": r(H, CK, sc=0.5), "out": r(H, H),
                                "gu": r(2 * I, H), "down": r(H, I)}),
              dict(**{"in": r(3 * H, H), "conv": r(H, CK, sc=0.5), "out": r(H, H)},
                   moe=dict(router=r(NE, H, sc=1.0), w13=r(NE, 2 * IE, H, sc=0.3), w2=r(NE, H, IE, sc=0.3))),
              dict(qkv=r((NH + 2 * NKV) * D, H), o=r(H, NH * D),
                   moe=dict(router=r(NE, H, sc=1.0), w13=r(NE, 2 * IE, H, sc=0.3), w2=r(NE, H, IE, sc=0.3)))]
    head = r(VPAD, H)
    head[V:] = 0
    return dict(emb=r(VPAD, H, sc=1.0), head=head, layers=layers), g


def build():
    ws, g = weights()
    return LM(ws, g)


def targets_for(lm):
    return M.DEFAULT_TARGETS


def test_check_supported_accepts_lfm_layout():
    M.check_supported(build())


def test_check_supported_still_refuses_grouped_routing_and_unknown_recurrence():
    lm = build()
    lm.model.layers[1].feed_forward.experts.routed_experts.num_expert_group = 2
    with pytest.raises(NotImplementedError, match="routing"):
        M.check_supported(lm)
    lm = build()
    bad = nn.Module()
    bad.get_state_shape = lambda: None
    lm.model.layers[0].add_module("mystery", bad)
    with pytest.raises(NotImplementedError, match="recurrent"):
        M.check_supported(lm)


def test_default_targets_cover_every_lfm_linear():
    lm = build()
    sft = T.sft_for(lm)
    names = set(sft.adapters)
    for want in ("model.layers.0.short_conv.in_proj", "model.layers.0.short_conv.out_proj",
                 "model.layers.0.feed_forward.w13", "model.layers.0.feed_forward.w2",
                 "model.layers.2.self_attn.qkv_proj", "model.layers.2.self_attn.out_proj"):
        assert want in names
    assert not any("experts" in n or n.endswith(".gate") for n in names)    # routed experts / router stay frozen


def test_trainmode_matches_oracle_and_restores():
    lm = build()
    sft = T.sft_for(lm)
    T.set_adapters(sft)
    want_loss, want = T.oracle(lm, sft)
    loss, got = T.run_micro(sft)
    assert abs(loss - want_loss) < 1e-4 * want_loss
    for n in want:
        for i, f in enumerate("AB"):
            torch.testing.assert_close(got[n][i], want[n][i], rtol=2e-3, atol=2e-5, msg=f"{n}.{f}")
    # every swap restored
    with pytest.raises(RuntimeError, match="state-cache"):
        lm.model.layers[0].short_conv(torch.zeros(2, H), torch.zeros(2, H))
    assert "forward" not in lm.model.layers[0].short_conv.__dict__ and M._TRAIN is None


def test_conv_state_starts_at_zero_per_segment():
    """Two packed examples must not see each other through the conv window."""
    lm = build()
    sft = T.sft_for(lm)
    T.set_adapters(sft)
    la, _ = T.run_micro(sft, dict(ids=T.BATCH["ids"][:10], seg_lens=[10], mask=T.BATCH["mask"][:10]))
    sft.abort()
    lb, _ = T.run_micro(sft, dict(ids=T.BATCH["ids"][10:], seg_lens=[7], mask=T.BATCH["mask"][10:]))
    sft.abort()
    lp, _ = T.run_micro(sft)
    assert abs(la + lb - lp) < 1e-4 * lp


def test_router_weights_follow_serving_selection_and_carry_gradient():
    """ids from the biased scores, weights from the unbiased ones, renormalized,
    scaled; the weights have a gradient back to the gate input."""
    lm = build()
    blk = lm.model.layers[1].feed_forward
    x = torch.randn(9, H, requires_grad=True)
    # bias large enough to change the selection vs no bias
    blk.gate.e_score_correction_bias.data = torch.tensor([2.0, -2.0, 0.0, 0.0])

    class S:   # TrainMode.route only needs nothing from the session
        pass
    tw, ids = M.TrainMode.route(S(), blk, blk.experts.routed_experts, x)
    score = torch.sigmoid(blk.gate(x)[0].float())
    want_ids = (score + blk.gate.e_score_correction_bias).topk(TOPK, -1).indices
    assert torch.equal(ids, want_ids)
    want = score.gather(-1, want_ids)
    want = want / want.sum(-1, keepdim=True) * SCALE
    torch.testing.assert_close(tw, want, rtol=1e-5, atol=1e-6)
    tw.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_runner_side_scaling_is_applied_once():
    lm = build()
    blk = lm.model.layers[1].feed_forward
    x = torch.randn(5, H)
    tw0, _ = M.TrainMode.route(object(), blk, blk.experts.routed_experts, x)
    blk.experts.routed_experts.routed_scaling_factor, blk.experts.routed_scaling_factor = 1.0, SCALE
    tw1, _ = M.TrainMode.route(object(), blk, blk.experts.routed_experts, x)
    torch.testing.assert_close(tw0, tw1)


def test_training_lowers_loss():
    lm = build()
    sft = T.sft_for(lm)
    sft.begin(dict(lr=3e-2))
    losses = []
    for _ in range(8):
        losses.append(sft.micro(dict(T.BATCH, norm=12.0))["loss_sum"] / 12.0)
        sft.step()
    sft.commit()
    assert losses[-1] < losses[0] - 0.05
