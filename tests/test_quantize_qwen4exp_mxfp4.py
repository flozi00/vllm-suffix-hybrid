# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp block-FP8 -> Quark MXFP4 converter (suffix_hybrid.tools.quantize_qwen4exp_mxfp4):
rounding bit-exact vs amd/Qwen3.8-Flash-Next-Quark-MXFP4, the layer policy, the Quark
exclude list as vLLM matches it (target model + MTP draft names), and an end-to-end
write of a tiny synthetic FP8 checkpoint (that part needs safetensors)."""
import json
import re

import pytest
import torch
from suffix_hybrid.tools import quantize_qwen4exp_mxfp4 as q

# 32 BF16 weights (little-endian hex) from Qwen/Qwen3.8-Flash-Next (fused
# experts.gate_up_proj / experts.down_proj rows) and the bytes that
# amd/Qwen3.8-Flash-Next-Quark-MXFP4 stores for them: 16 packed bytes of
# `<expert>.weight` and the `<expert>.weight_scale` byte. The first block of each
# pair has an amax mantissa >= 1.75 (Quark's "even" rounding raises the shared
# exponent where floor(log2(amax)) would not); every block has a negative value
# that rounds to zero, which Quark writes as +0.
KAT = [
    ("L0.E0.gate_proj.row0.blk14",
     "963ccdbb68bc14bc6abca2bb40bc863aec3c4ebbe03b8ebb5a3ce23c113ca83b"
     "683b2b3bff3b2cbbe33b073be0ba9a3c9cbba13b943b3c3c3fbbe73cffbb9dbc",
     "a4ac9c0b9692631211921240193169ca", 120),
    ("L0.E0.gate_proj.row0.blk1",
     "8dbc9bbcb1bcb63bcdb9543c5f3b07bcd2bb4ebb6c3cf7ba27bc253c073c713b"
     "9a3c603cc33c6339f43b693c84bb263c29bc223cf33be13b1c3c8ebcdc3bafbb",
     "ee3f50c2ab965d246607645a5d44e4b3", 119),
    ("L17.E300.up_proj.row3.blk0",
     "a53ccbbb143c2ebc643c83bca4bc89bcffbca83c023c2a3b0b3b823cddbca6bc"
     "b23ceb3ad03a323ce6bbe83b01ba4abb15bc89b89fbc593c8abb13bc0dba98bc",
     "a5b2c4cd5e1241dd05302a900a3ca9c0", 120),
    ("L17.E300.up_proj.row3.blk2",
     "c9ba3a3a5fbb0d3bbebc803c623cdabb87bb4ebbeabb30bc4cbb123ca0bcb8ba"
     "c8bc5bbbdfbb69390d3c8bbc75bc9b3ca8bc3bbc033c3f3c23bc683c7cb9173c",
     "091a6fb6aadc4a9eaf0be46edf546d40", 119),
    ("L47.E511.down_proj.row5.blk0",
     "3cbcf83ce73a85ba313cb83c203b833c883c62bc88bc373c16bb8b3cbebca43c"
     "b63a5ebc26bb263addbbeeb8233ce73c633c3fbc293c87bc8e3b853b443b80bc",
     "6b005341c43c495db0090a63b4c311c1", 120),
    ("L47.E511.down_proj.row5.blk11",
     "47bc633b70bcc7bc973ced3a0abcaf3b863b023c8f3cddbc78ba7a3ca73c8e3a"
     "94bc87bc453c90bca6bb7e3c91bc9abce4bb913cda3c70bc133b803ccfbc573b",
     "2dfe163c42f66017eee56bee6ce7612f", 119),
]


def _bf16(hexstr):
    raw = torch.frombuffer(bytearray.fromhex(hexstr), dtype=torch.int16)
    return raw.view(torch.bfloat16).reshape(1, -1)


@pytest.mark.parametrize("src,bf16,packed,scale", KAT, ids=[k[0] for k in KAT])
def test_bit_exact_vs_amd_quark_checkpoint(src, bf16, packed, scale):
    p, s = q.mxfp4_quantize(_bf16(bf16))
    assert p.dtype == torch.uint8 and s.dtype == torch.uint8
    assert bytes(p.flatten().tolist()).hex() == packed
    assert s.tolist() == [[scale]]


def _codes(packed):
    return torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).tolist()[0]


def test_rounding_ties_saturation_zero_and_nibble_order():
    # amax 6 -> shared exponent 0 (scale byte 127): values are their own e2m1 inputs
    row = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.3, 6.0, 0.2, -6.0]
    p, s = q.mxfp4_quantize(torch.tensor([row + [0.0] * (32 - len(row))]))
    assert s.tolist() == [[127]]
    # ties go to the even code; -0.25 -> +0 (not 0x8); -0.3 -> -0.5 (0x9)
    assert _codes(p)[:len(row)] == [0, 2, 2, 4, 4, 6, 6, 0, 9, 7, 0, 15]
    assert int(p[0, 0]) == 0 | (2 << 4)  # element 0 in the low nibble
    # mantissa 1.725 < 1.75: exponent stays 0 and 6.9 saturates to 6
    p, s = q.mxfp4_quantize(torch.tensor([[6.9] + [0.0] * 31]))
    assert s.tolist() == [[127]] and _codes(p)[0] == 7
    # mantissa 1.75: Quark raises the exponent (floor(log2(7)) - 2 would give 0)
    p, s = q.mxfp4_quantize(torch.tensor([[7.0, 1.0] + [0.0] * 30]))
    assert s.tolist() == [[128]] and _codes(p)[:2] == [6, 1]  # 7/2 = 3.5 -> tie -> 4
    # all-zero block: smallest e8m0 exponent, all codes +0
    p, s = q.mxfp4_quantize(torch.zeros(2, 64))
    assert s.tolist() == [[0, 0], [0, 0]] and p.eq(0).all()
    with pytest.raises(ValueError):
        q.mxfp4_quantize(torch.zeros(1, 48))
    with pytest.raises(ValueError):
        q.mxfp4_quantize(torch.tensor([[float("nan")] + [0.0] * 31]))


def test_roundtrip_error_and_row_chunks():
    torch.manual_seed(0)
    w = torch.randn(96, 256) * 0.02
    p, s = q.mxfp4_quantize(w)
    assert p.shape == (96, 128) and s.shape == (96, 8)
    rel = float((q.mxfp4_dequantize(p, s) - w).norm() / w.norm())
    assert 0.05 < rel < 0.13
    pc, sc = q.quantize_rows(w, rows=40)  # chunked == whole
    assert torch.equal(pc, p) and torch.equal(sc, s)


@pytest.mark.parametrize("n,k", [(256, 384), (200, 300)])
def test_fp8_block_dequant(n, k):
    torch.manual_seed(1)
    w8 = (torch.randn(n, k) * 4).to(torch.float8_e4m3fn)
    sinv = (torch.rand(-(-n // 128), -(-k // 128)) + 0.5).to(torch.bfloat16)
    ref = torch.empty(n, k)
    for i in range(n):
        for j in range(0, k, 64):
            ref[i, j:j + 64] = w8[i, j:j + 64].float() * sinv[i // 128, j // 128].float()
    assert torch.equal(q.dequant_fp8_block(w8, sinv), ref)
    with pytest.raises(ValueError):
        q.dequant_fp8_block(w8, sinv[:1])


T = "model.language_model.layers.7."


@pytest.mark.parametrize("key,mx,bf", [
    (T + "mlp.experts.511.gate_proj.weight", "expert", "expert"),
    (T + "mlp.experts.0.down_proj.weight", "expert", "expert"),
    ("mtp.layers.0.mlp.experts.3.up_proj.weight", "expert", "expert"),
    (T + "mlp.shared_expert.gate_proj.weight", "mxfp4", "mxfp4"),
    ("mtp.layers.0.mlp.shared_expert.down_proj.weight", "mxfp4", "mxfp4"),
    (T + "linear_attn.out_proj.weight", "mxfp4", "mxfp4"),
    (T + "self_attn.o_proj.weight", "mxfp4", "mxfp4"),
    (T + "linear_attn.in_proj_qkv.weight", "mxfp4", "copy"),
    (T + "linear_attn.in_proj_z.weight", "mxfp4", "copy"),
    (T + "self_attn.q_proj.weight", "mxfp4", "copy"),
    (T + "self_attn.v_proj.weight", "mxfp4", "copy"),
    (T + "linear_attn.in_proj_a.weight", "copy", "copy"),
    (T + "linear_attn.in_proj_b.weight", "copy", "copy"),
    (T + "linear_attn.conv1d.weight", "copy", "copy"),
    (T + "linear_attn.A_log", "copy", "copy"),
    (T + "self_attn.indexer.index_qk_proj.weight", "copy", "copy"),
    (T + "mlp.gate.weight", "copy", "copy"),
    (T + "mlp.shared_expert_gate.weight", "copy", "copy"),
    (T + "attn_hyper_connection.input_mix_weight_down.weight", "copy", "copy"),
    ("model.language_model.layers.1.ple.key_proj.weight", "copy", "copy"),
    ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight", "copy", "copy"),
    ("mtp.layers.0.self_attn.q_proj.weight", "copy", "copy"),
    ("mtp.fc_hidden.weight", "copy", "copy"),
    ("model.visual.blocks.0.attn.qkv.weight", "copy", "copy"),
    ("lm_head.weight", "copy", "copy"),
    ("model.language_model.embed_tokens.weight", "copy", "copy"),
])
def test_layer_policy(key, mx, bf):
    assert q.classify(key, "mxfp4") == mx and q.classify(key, "bf16") == bf


def test_quark_config_schema():
    c = q.quark_config()
    assert c["quant_method"] == "quark" and "export" in c and c["layer_quant_config"] == {}
    assert c["layer_type_quant_config"] == {}  # vLLM .get()s it as a dict
    g = c["global_quant_config"]
    for part, dynamic in (("weight", False), ("input_tensors", True)):
        assert g[part]["dtype"] == "fp4" and g[part]["qscheme"] == "per_group"
        assert g[part]["group_size"] == 32 and g[part]["scale_format"] == "e8m0"
        assert g[part]["scale_calculation_mode"] == "even" and g[part]["is_dynamic"] is dynamic
    assert set(q.quark_config("bf16")["exclude"]) - set(c["exclude"]) == set(q.EXCLUDE_RISKY_BF16)
    with pytest.raises(ValueError):
        q.quark_config("fp8")


# --- the exclude list as vLLM evaluates it ------------------------------------
# vllm/model_executor/layers/quantization/quark/utils.py should_ignore_layer +
# utils/config_utils.py find_matching_patterns: exact name or "re:" + re.match; a
# fused module counts as excluded when every shard is; routed experts are also
# excluded by any non-regex entry naming one of their children.
PACKED = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"],
          "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
          "kv_proj": ["key_proj", "value_proj"]}


def _match(name, pattern):
    return bool(re.match(pattern[3:], name)) if pattern.startswith("re:") else name == pattern


def _vllm_excluded(name, exclude, experts=False):
    if experts and any(t == name or t.startswith(name + ".") for t in exclude
                       if not t.startswith("re:")):
        return True
    if any(_match(name, p) for p in exclude):
        return True
    proj = name.split(".")[-1]
    if proj not in PACKED:
        return False
    hits = [any(_match(name.replace(proj, s), p) for p in exclude) for s in PACKED[proj]]
    assert all(hits) or not any(hits), f"{name}: shards disagree"
    return all(hits)


def _modules():
    """(vLLM module name, its checkpoint weight names, is routed experts).

    Target names are what vLLM sees after the Qwen3.5/Qwen3-VL mapper
    (model.language_model. -> language_model.model., model.visual. -> visual.).
    The MTP draft is built with prefix "mtp" and its layer at index
    num_hidden_layers (48), while the checkpoint says mtp.layers.0."""
    rows = []
    for vl, hf in (("language_model.model.layers.5", "model.language_model.layers.5"),
                   ("mtp.layers.48", "mtp.layers.0")):
        rows += [
            (f"{vl}.self_attn.qkv_proj", [f"{hf}.self_attn.{p}_proj.weight" for p in "qkv"], False),
            (f"{vl}.self_attn.o_proj", [f"{hf}.self_attn.o_proj.weight"], False),
            (f"{vl}.self_attn.indexer.index_qk_proj",
             [f"{hf}.self_attn.indexer.index_qk_proj.weight"], False),
            (f"{vl}.mlp.experts", [f"{hf}.mlp.experts.{e}.{p}_proj.weight"
                                   for e in (0, 511) for p in ("gate", "up", "down")], True),
            (f"{vl}.mlp.shared_expert.gate_up_proj",
             [f"{hf}.mlp.shared_expert.{p}_proj.weight" for p in ("gate", "up")], False),
            (f"{vl}.mlp.shared_expert.down_proj", [f"{hf}.mlp.shared_expert.down_proj.weight"], False),
        ]
    t, h = "language_model.model.layers.4", "model.language_model.layers.4"
    rows += [
        (f"{t}.linear_attn.in_proj_qkvz",
         [f"{h}.linear_attn.in_proj_qkv.weight", f"{h}.linear_attn.in_proj_z.weight"], False),
        (f"{t}.linear_attn.in_proj_ba",
         [f"{h}.linear_attn.in_proj_b.weight", f"{h}.linear_attn.in_proj_a.weight"], False),
        (f"{t}.linear_attn.out_proj", [f"{h}.linear_attn.out_proj.weight"], False),
        ("language_model.model.layers.1.ple.kv_proj",
         ["model.language_model.layers.1.ple.key_proj.weight",
          "model.language_model.layers.1.ple.value_proj.weight"], False),
        ("visual.blocks.3.attn.qkv", ["model.visual.blocks.3.attn.qkv.weight"], False),
        ("visual.merger.linear_fc2", ["model.visual.merger.linear_fc2.weight"], False),
        ("mtp.fc_embedding", ["mtp.fc_embedding.weight"], False),
        ("mtp.fc_hidden", ["mtp.fc_hidden.weight"], False),
    ]
    return rows


@pytest.mark.parametrize("risky", q.RISKY_CHOICES)
def test_exclude_list_agrees_with_written_tensors(risky):
    """vLLM quantizes a module exactly when the converter wrote MXFP4 for all of its
    checkpoint weights; anything excluded must have been copied unchanged."""
    exclude = q.quark_config(risky)["exclude"]
    for name, weights, experts in _modules():
        kinds = {q.classify(w, risky) for w in weights}
        assert len(kinds) == 1, (name, kinds)
        excluded = _vllm_excluded(name, exclude, experts)
        assert excluded == (kinds == {"copy"}), (risky, name, kinds)


def test_mtp_draft_names():
    ex = q.quark_config()["exclude"]
    assert _vllm_excluded("mtp.layers.48.self_attn.qkv_proj", ex)
    assert _vllm_excluded("mtp.fc_hidden", ex) and _vllm_excluded("mtp.fc_embedding", ex)
    assert not _vllm_excluded("mtp.layers.48.mlp.experts", ex, experts=True)
    assert not _vllm_excluded("mtp.layers.48.mlp.shared_expert.down_proj", ex)
    # the target's own attention stays quantized (the MTP rule is anchored at ^mtp)
    assert not _vllm_excluded("language_model.model.layers.11.self_attn.o_proj", ex)


# --- end to end on a tiny synthetic FP8 checkpoint ----------------------------
def _fp8_expert(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    w8 = (torch.randn(n, k, generator=g) * 8).to(torch.float8_e4m3fn)
    sinv = (torch.rand(-(-n // 128), -(-k // 128), generator=g) * 1e-3 + 1e-4).to(torch.bfloat16)
    return w8, sinv


def _tiny_fp8_checkpoint(d):
    pytest.importorskip("safetensors")  # in the vLLM image; not in the CPU CI venv
    from safetensors.torch import save_file
    torch.manual_seed(2)

    def bf(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.05

    L, M = "model.language_model.layers.0.", "mtp.layers.0."
    a, b = {}, {}
    seed = 0
    for pre, shard in ((L, a), (M, b)):
        for e in range(2):
            for proj, (n, k) in (("gate", (64, 256)), ("up", (64, 256)), ("down", (256, 64))):
                w8, sinv = _fp8_expert(n, k, seed)
                seed += 1
                key = f"{pre}mlp.experts.{e}.{proj}_proj.weight"
                shard[key] = w8
                # one scale lives in the other shard: the lookup must cross shards
                (b if (pre, e, proj) == (L, 1, "down") else shard)[key + "_scale_inv"] = sinv
        for proj, shape in (("gate", (64, 256)), ("up", (64, 256)), ("down", (256, 64))):
            shard[f"{pre}mlp.shared_expert.{proj}_proj.weight"] = bf(*shape)
        shard[f"{pre}mlp.gate.weight"] = bf(2, 256)
        shard[f"{pre}mlp.shared_expert_gate.weight"] = bf(1, 256)
        for p in ("q", "k", "v", "o"):
            shard[f"{pre}self_attn.{p}_proj.weight"] = bf(64, 256)
    a.update({
        L + "linear_attn.in_proj_qkv.weight": bf(96, 256),
        L + "linear_attn.in_proj_z.weight": bf(64, 256),
        L + "linear_attn.out_proj.weight": bf(256, 64),
        L + "linear_attn.in_proj_a.weight": bf(8, 256),
        L + "linear_attn.conv1d.weight": bf(96, 1, 4),
        L + "attn_hyper_connection.input_mix_weight_down.weight": bf(32, 1024),
        L + "ple.key_proj.weight": bf(1024, 256),
        L + "ple.ple_embedding.ngram_embedding.shard_0.weight":
            (torch.randn(10, 32) * 50).to(torch.float8_e4m3fn),
        L + "ple.ple_embedding.ngram_embedding.weight_scale": torch.tensor([0.01], dtype=torch.bfloat16),
        L + "ple.ple_embedding.layer_multipliers": torch.tensor([3, 5, 7]),
        "model.language_model.embed_tokens.weight": bf(16, 256),
        "lm_head.weight": bf(16, 256),
        "model.visual.blocks.0.attn.qkv.weight": bf(48, 32),
    })
    b["mtp.fc_hidden.weight"] = bf(256, 256)
    save_file(a, str(d / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(b, str(d / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    wm = {k: "model-00001-of-00002.safetensors" for k in a}
    wm.update({k: "model-00002-of-00002.safetensors" for k in b})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (d / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
                                "weight_block_size": [128, 128],
                                "modules_to_convert": ["ple.ple_embedding.ngram_embedding"]},
        "text_config": {"num_hidden_layers": 1, "mtp_num_hidden_layers": 1, "num_experts": 2}}))
    (d / "tokenizer_config.json").write_text("{}")
    (d / "chat_template.jinja").write_text("{{ x }}")
    (d / ".complete.json").write_text("{}")  # weight-staging marker: never copied
    return {**a, **b}


@pytest.mark.parametrize("risky,workers", [("mxfp4", 1), ("bf16", 2)])
def test_end_to_end(tmp_path, capsys, risky, workers):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    orig = _tiny_fp8_checkpoint(src)
    assert q.main(["--src", str(src), "--out", str(out), "--risky", risky,
                   "--workers", str(workers)]) == 0
    log = capsys.readouterr().out
    assert q.DONE_MARKER in log
    assert sorted(p.name for p in out.iterdir()) == [
        "chat_template.jinja", "config.json", "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors", "model.safetensors.index.json",
        q.MANIFEST_NAME, "tokenizer_config.json"]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"] == q.quark_config(risky)
    assert cfg["text_config"]["ple_embedding_dtype"] == "float8_e4m3fn"
    man = json.loads((out / q.MANIFEST_NAME).read_text())
    n_dense = 2 + 2 * 3 + (5 if risky == "mxfp4" else 0)  # o_proj, out_proj; shared; risky
    assert man["stats"]["tensors"] == {"expert": 12, "mxfp4": n_dense,
                                       "copy": len(orig) - 12 - 12 - n_dense}
    assert set(man["source_sha256"]) == {"config.json", "model.safetensors.index.json",
                                         "model-00001-of-00002.safetensors",
                                         "model-00002-of-00002.safetensors"}
    from safetensors import safe_open
    wm = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    seen = {}
    for name in sorted(set(wm.values())):
        with safe_open(str(out / name), "pt") as st:
            for key in st.keys():
                seen[key] = st.get_tensor(key)
    assert set(seen) == set(wm) and not any(k.endswith("_scale_inv") for k in seen)
    key = "model.language_model.layers.0.mlp.experts.1.down_proj.weight"  # cross-shard scale
    p, s = q.mxfp4_quantize(q.dequant_fp8_block(orig[key], orig[key + "_scale_inv"]))
    assert seen[key].dtype == torch.uint8 and seen[key].shape == (256, 32)
    assert torch.equal(seen[key], p) and torch.equal(seen[key + "_scale"], s)
    key = "mtp.layers.0.mlp.experts.0.gate_proj.weight"
    assert seen[key].shape == (64, 128) and seen[key + "_scale"].shape == (64, 8)
    key = "mtp.layers.0.mlp.shared_expert.down_proj.weight"
    assert seen[key].dtype == torch.uint8 and torch.equal(seen[key], q.mxfp4_quantize(orig[key])[0])
    for key in ("model.language_model.layers.0.self_attn.o_proj.weight",
                "model.language_model.layers.0.linear_attn.out_proj.weight"):
        assert seen[key].dtype == torch.uint8
    for key in ("model.language_model.layers.0.self_attn.q_proj.weight",
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"):
        assert (seen[key].dtype == torch.uint8) == (risky == "mxfp4")
    for key, t in orig.items():  # everything else is copied byte-identical
        if q.classify(key, risky) == "copy" and not key.endswith("_scale_inv"):
            assert seen[key].dtype == t.dtype and torch.equal(seen[key].view(torch.uint8),
                                                              t.view(torch.uint8)), key


def test_refuses_nonempty_out_and_wrong_source(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _tiny_fp8_checkpoint(src)
    out.mkdir()
    (out / "x").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        q.main(["--src", str(src), "--out", str(out)])
    cfg = json.loads((src / "config.json").read_text())
    cfg["quantization_config"]["weight_block_size"] = [1, 128]
    (src / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="block-FP8"):
        q.main(["--src", str(src), "--out", str(tmp_path / "out2")])
