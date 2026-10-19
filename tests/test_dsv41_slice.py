# SPDX-License-Identifier: Apache-2.0
"""CPU tests for suffix_hybrid/tools/dsv41_slice.py: everything that needs no GPU and no vLLM.

What these prove: the per-rank derivation (shapes vs the prod rank, to the byte), the Engram table
arithmetic (reproduces the real engram_num_embeddings and the logged per-rank rows), opts validation,
the engine kwargs, the dry-run CLI, the pure measurement helpers. What they cannot prove: that vLLM
boots this config (see docs/dsv41-slice.md, "Unverified").
"""
from __future__ import annotations

import hashlib
import io
import json
import os

import pytest

from suffix_hybrid.tools import dsv41_slice as d

REAL = d.REAL_CONFIG["text_config"]


def opts(**kw):
    return d.parse_opts(kw)


def fails(code=2, **kw):
    with pytest.raises(d.SliceError) as ei:
        d.parse_opts(kw)
    assert ei.value.code == code
    return str(ei.value)


# ---------------------------------------------------------------- embedded config
def test_embedded_config_is_the_hf_file():
    assert hashlib.sha256(d._REAL_CONFIG_JSON.encode()).hexdigest() == d.REAL_CONFIG_SHA256
    assert d.REAL_CONFIG_SHA256 == "fbb1660e71f4d05dad68aa1849188791a1e80a8b1d2a031ce543ff24c65a448e"
    # json.dumps(indent=2) + newline reproduces the file byte for byte, so the dict IS the file
    assert (json.dumps(d.REAL_CONFIG, indent=2) + "\n") == d._REAL_CONFIG_JSON
    assert d.REAL_CONFIG["architectures"] == ["DeepseekV41ForCausalLM"]


def test_real_config_is_returned_as_a_copy():
    c = d.real_config()
    c["text_config"]["num_hidden_layers"] = 1
    assert d.REAL_CONFIG["text_config"]["num_hidden_layers"] == 40


def test_layer_lists_and_full_depth():
    t = REAL
    ratios = t["compress_ratios"]
    assert t["num_hidden_layers"] == 40 and len(ratios) == 43  # 40 backbone + 3 DSpark layers
    assert ratios[:40].count(0) == 2 and ratios[:40].count(2) == 18 and ratios[:40].count(1) == 20
    assert ratios[40:] == [0, 0, 0]
    assert t["index_source_layer_ids"] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert t["kv_source_layer_ids"] == [2, 8, 14, 20]
    assert t["engram_layer_ids"] == [1, 14] and t["dspark_target_layer_ids"] == [37, 38, 39]
    sl = d.slice_config(opts())[0]["text_config"]
    for key in ("num_hidden_layers", "compress_ratios", "index_source_layer_ids", "kv_source_layer_ids",
                "engram_layer_ids", "dspark_target_layer_ids", "num_nextn_predict_layers", "candidate_source_layer_id",
                "candidate_topk_blocks", "sliding_window", "hc_mult", "moe_intermediate_size", "n_shared_experts"):
        assert sl[key] == t[key], key


# ---------------------------------------------------------------- Engram arithmetic
def test_engram_primes_reproduce_the_real_table_rows():
    rows = d.engram_rows(REAL["engram_layer_ids"], REAL["engram_max_ngram_size"], REAL["engram_n_heads"],
                         REAL["engram_vocab_size"])
    assert rows == REAL["engram_num_embeddings"] == [384006168, 384016682]


def test_engram_per_rank_rows_match_the_prod_boot_log():
    # boot5.log: TP0 offloads 48000217 rows (layer 1) and 48001463 rows (layer 14) = 3 of the 24 columns
    primes = d.engram_primes([1, 14], 4, 8, 16000000)
    cols = [[p for g in layer for p in g] for layer in primes]
    assert sum(cols[0][:3]) == 48000217
    assert sum(cols[1][:3]) == 48001463
    # 3 columns x 136 B (MXFP4 row + scales) = the logged 6.08 GiB per rank and table
    assert round(48000217 * (128 + 8) / 2 ** 30, 2) == 6.08


def test_engram_shrink_defaults():
    o = opts()
    t = d.slice_config(o)[0]["text_config"]
    assert t["engram_n_heads"] == 1 and t["engram_vocab_size"] == 1048576
    prime_rows = d.engram_rows([1, 14], 4, 1, 1048576)
    assert t["engram_num_embeddings"] == prime_rows and all(3 * 1048576 <= r < 3 * 1048576 + 200 for r in prime_rows)
    assert t["engram_compressed_vocab_size"] == t["vocab_size"] == 16160
    assert d.engram_host_gib(o) == pytest.approx(2 * 3 * 1048576 * 136 / 2 ** 30)  # 0.8 GiB pinned


def test_engram_all_24_columns_variant_and_host_cap():
    o = opts(engram_heads=8, engram_rows=1000000)
    t = d.slice_config(o)[0]["text_config"]
    assert t["engram_n_heads"] == 8 and len(d.engram_primes([1, 14], 4, 8, 1000000)[0][0]) == 8
    assert "pin" in fails(engram_heads=8, engram_rows=16000000)  # 24 cols x 16M rows x 136 B x 2 layers


# ---------------------------------------------------------------- per-rank derivation
def test_slice_config_dimensions_are_one_rank():
    t = d.slice_config(opts())[0]["text_config"]
    assert (t["num_attention_heads"], t["o_groups"]) == (8, 1)
    assert (t["n_routed_experts"], t["num_experts_per_tok"]) == (48, 1)
    assert t["moe_intermediate_size"] == 2304  # EP: whole experts at the full width
    assert t["vocab_size"] == 16160 == 129280 // 8
    assert (t["dspark_n_routed_experts"], t["dspark_num_experts_per_tok"]) == (16, 1)
    assert 0 <= t["dspark_noise_token_id"] < t["vocab_size"]
    assert t["dspark_noise_token_id"] == 128799 - 7 * 16160  # local index on the owning shard


def test_slice_config_changes_only_documented_keys():
    cfg, rules = d.slice_config(opts())
    diff = d.config_diff(d.REAL_CONFIG, cfg)
    assert set(diff) == {"text_config." + k for k in (
        "num_attention_heads", "o_groups", "n_routed_experts", "num_experts_per_tok", "vocab_size",
        "dspark_n_routed_experts", "dspark_num_experts_per_tok", "dspark_noise_token_id", "engram_n_heads",
        "engram_vocab_size", "engram_num_embeddings", "engram_compressed_vocab_size")}
    assert {r["key"] for r in rules} == {k.split(".", 1)[1] for k in diff}
    for key in ("quantization_config", "vision_config", "architectures", "model_type", "dtype"):
        assert cfg[key] == d.REAL_CONFIG[key], key  # the same quant methods and the VL arch are instantiated
    assert cfg["quantization_config"]["engram_embed_quant"] == "mxfp4"  # the plugin overlay keys off this


def test_slice_refuses_a_layout_that_does_not_divide():
    with pytest.raises(d.SliceError):
        d.slice_text(REAL, opts(), tp=3, ep=8)


def test_rank_equivalence_is_exact_except_the_documented_set():
    real = d.real_config()
    sl = d.slice_config(opts())[0]
    mr = d.rank_modules(real, 8, 8)
    ms = d.rank_modules(sl, 1, 1, shared_tp=8)  # the shared-expert patch emulates TP8 width
    diff = d.rank_diff(mr, ms)
    assert set(diff) == {"moe.gate", "draft.gate", "draft.markov_w1", "draft.markov_w2", "embed", "lm_head"}
    # lm_head / embedding differ only by vocab padding (16192 vs 16160 rows)
    assert diff["lm_head"][1] / diff["lm_head"][0] == pytest.approx(16192 / 16160)
    # the per-rank GEMM shapes themselves are identical
    assert ms["attn.wq_b"]["elems"] == mr["attn.wq_b"]["elems"] == 1280 * 4096
    assert ms["attn.wo_a"]["elems"] == mr["attn.wo_a"]["elems"] == 4096 * 1024
    assert ms["attn.wo_b"]["elems"] == mr["attn.wo_b"]["elems"] == 1024 * 5120
    assert ms["moe.experts"]["elems"] == mr["moe.experts"]["elems"] == 48 * 3 * 5120 * 2304
    assert ms["engram.wkv"]["elems"] == mr["engram.wkv"]["elems"] == 768 * 25600 == 6144 * 3200
    assert ms["moe.shared"]["elems"] == mr["moe.shared"]["elems"] == 3 * 5120 * 288


def test_shared_expert_is_8x_without_the_patch():
    sl = d.slice_config(opts())[0]
    unpatched = d.rank_modules(sl, 1, 1)
    patched = d.rank_modules(sl, 1, 1, shared_tp=8)
    assert unpatched["moe.shared"]["elems"] == 8 * patched["moe.shared"]["elems"]
    assert "moe.shared" in d.rank_diff(d.rank_modules(d.real_config(), 8, 8), unpatched)


def test_weight_model_is_anchored_to_the_logged_40_66_gib():
    real = d.resident_bytes(d.rank_modules(d.real_config(), 8, 8)) / 2 ** 30
    # vLLM logged "Model loading took 40.66 GiB" on glm; the model leaves out the vision tower (0.6-1.1),
    # rope caches and buffers, so it must land a little below
    assert 37.5 < real < 40.66
    sl = d.resident_bytes(d.rank_modules(d.slice_config(opts())[0], 1, 1, shared_tp=8)) / 2 ** 30
    assert abs(sl - real) < 0.5


def test_expert_streaming_matches_the_comms_doc_table():
    # docs/dsv41-comms.md section 2: EP mean MB per rank per layer at T = 6/12/48/96/192 tokens
    per_expert = 3 * 5120 * 2304 * d.BPE["nvfp4"] / 1e6
    assert per_expert == pytest.approx(19.9, abs=0.05)
    for tokens, mb in ((6, 86), (12, 164), (48, 507), (96, 745), (192, 909)):
        got = d.touched_experts(tokens, 6, 384, 8) * per_expert
        assert got == pytest.approx(mb, abs=2.5), tokens


def test_step_bytes_roofline_matches_the_task_estimate():
    # lane brief: per-GPU weight streaming ~22.6 ms at c=32 at the 1.79 TB/s peak
    sb = d.step_bytes(d.real_config(), 8, 8, 32, 5)
    assert 21.0 < sb["total"] / 1.79e12 * 1e3 < 24.0
    assert d.step_bytes(d.real_config(), 8, 8, 1, 0)["draft_experts"] == 0.0  # k=0: no draft traffic


# ---------------------------------------------------------------- bias / non-equivalences
def test_bias_terms_signs_and_magnitudes():
    o = opts()
    for c in (1, 8, 32):
        t = d.bias_terms(o, c)
        assert t["collectives"] < 0 and t["moe_pairs"] > 0 and t["shared_expert"] == 0
        assert t["markov_head"] < 0 and t["sampler_vocab"] < 0 and t["gate_width"] < 0
        assert t["quantified_total"] == pytest.approx(sum(v for k, v in t.items() if k != "quantified_total"), abs=1e-3)
    assert d.bias_terms(o, 8)["moe_pairs"] > d.bias_terms(o, 32)["moe_pairs"]  # the c=8 over-count is the biggest
    assert d.bias_terms(opts(shared_expert_tp=False), 1)["shared_expert"] == pytest.approx(0.98, abs=0.05)  # 7/8 of 196 MB at 1.4 TB/s
    assert d.bias_terms(opts(moe_topk=2), 8)["moe_pairs"] > d.bias_terms(opts(), 8)["moe_pairs"]


def test_prod_equivalent_adds_back_the_bias():
    o = opts()
    bias = d.bias_terms(o, 32)["quantified_total"]
    assert d.prod_equivalent_ms(o, 32, 30.0) == pytest.approx(30.0 - bias, abs=1e-3)
    assert d.prod_equivalent_ms(o, 32, 30.0) > 50  # the host all-reduce time is added back


def test_expected_slice_step_is_the_live_step_plus_the_bias():
    o = opts()
    for c in (1, 8, 32):
        assert d.expected_slice_ms(o, c) == pytest.approx(d.LIVE_STEP_MS[c] + d.bias_terms(o, c)["quantified_total"], abs=1e-3)
    assert [d.expected_slice_ms(o, c) for c in (1, 8, 32)] == pytest.approx([11.01, 21.56, 30.3], abs=0.05)
    assert d.expected_slice_ms(o, 4) is None  # no live reference at c=4
    assert d.expected_slice_ms(opts(k=3), 8) is None and d.expected_slice_ms(opts(enable_adaptive_verification=True), 8) is None


def test_non_equivalence_list_is_complete_and_keyed_by_c():
    ne = d.non_equivalences(opts(conc=[1, 8]))
    ids = [n["id"] for n in ne]
    assert ids[:2] == ["collectives", "moe_pairs"] and "shared_expert" in ids and "vision_tower" in ids
    assert all(set(n["ms"]) == {"1", "8"} for n in ne)
    assert all(n["basis"] and n["prod"] and n["slice"] for n in ne)


# ---------------------------------------------------------------- opts
def test_opts_defaults_are_the_prod_flags():
    o = opts()
    assert (o["max_model_len"], o["max_num_seqs"], o["max_num_batched_tokens"]) == (524288, 32, 8192)
    assert (o["block_size"], o["kv_cache_dtype"], o["k"], o["conc"]) == (64, "fp8", 5, [1, 8, 32])
    assert o["rejection_sample_method"] == "synthetic" and o["synthetic"] == "chat"
    assert o["gpu_memory_utilization"] == 0.93 and o["enable_prefix_caching"] is True
    assert d.parse_opts(None) == o


@pytest.mark.parametrize("key,bad", [
    ("prompt_tokens", 15), ("prompt_tokens", 262145), ("prompt_tokens", True), ("prompt_tokens", "2048"),
    ("prompt_tokens", 20.5), ("warmup_steps", -1), ("warmup_steps", 1001), ("measure_steps", 3),
    ("measure_steps", 4001), ("temperature", -0.1), ("temperature", 2.1), ("temperature", float("nan")),
    ("profile_steps", 0), ("profile_steps", 65), ("k", -1), ("k", 9), ("gpu_memory_utilization", 0.29),
    ("gpu_memory_utilization", 0.99), ("max_model_len", 4095), ("max_model_len", 1048577), ("max_num_seqs", 0),
    ("max_num_seqs", 257), ("max_num_batched_tokens", 511), ("max_num_batched_tokens", 131073),
    ("moe_topk", 0), ("moe_topk", 7), ("engram_rows", 1023), ("engram_rows", 16000001), ("seed", -1),
    ("seed", 2 ** 31), ("deadline_s", 59), ("deadline_s", 43201), ("conc", []), ("conc", [0]),
    ("conc", [257]), ("conc", [1, 1]), ("conc", list(range(1, 10))), ("conc", "8"), ("conc", [1.5]),
])
def test_opts_numeric_bounds(key, bad):
    with pytest.raises(d.SliceError) as ei:
        d.parse_opts({key: bad})
    assert ei.value.code == 2 and "opts:" in str(ei.value)


@pytest.mark.parametrize("key,bad", [
    ("draft_sample_method", "sampling"), ("rejection_sample_method", "typical"), ("block_size", 16),
    ("block_size", 128), ("kv_cache_dtype", "fp4"), ("cudagraph_mode", "full"), ("moe_backend", "magic"),
    ("linear_backend", "x"), ("sparse_indexer_topk_backend", "y"), ("moe_routing", "zipf"),
    ("engram_heads", 2), ("log_level", "TRACE"), ("profile", 1), ("shared_expert_tp", "yes"),
    ("enable_prefix_caching", None), ("async_scheduling", "auto"), ("work_dir", "relative/path"),
    ("synthetic", "nonsense"), ("synthetic", []), ("synthetic", [1.2]), ("env", ["A"]),
])
def test_opts_enums_and_types(key, bad):
    with pytest.raises(d.SliceError) as ei:
        d.parse_opts({key: bad})
    assert ei.value.code == 2


def test_opts_accept_the_boundaries():
    assert opts(prompt_tokens=16, warmup_steps=0, measure_steps=4, k=0, temperature=0.0)["k"] == 0
    assert opts(max_num_seqs=256, conc=[256], max_num_batched_tokens=131072)["conc"] == [256]
    assert opts(gpu_memory_utilization=0.3, engram_rows=1024)["engram_rows"] == 1024
    assert opts(conc=[32, 1, 8])["conc"] == [1, 8, 32]  # sorted
    assert opts(cudagraph_mode="FULL_AND_PIECEWISE", moe_backend="flashinfer_cutlass")["moe_backend"]


def test_unknown_key_is_refused_with_the_valid_list():
    msg = fails(kk=5)
    assert "unknown keys" in msg and "conc" in msg


def test_cross_checks():
    assert "max_num_seqs" in fails(conc=[64])
    assert "k=0" in fails(k=0, synthetic="chat")
    assert "k=0" in fails(k=0, enable_adaptive_verification=True)
    assert "k=0" in fails(k=0, draft_sample_method="probabilistic")
    assert opts(k=0)["rejection_sample_method"] is None
    assert "mutually exclusive" in fails(synthetic="chat", synthetic_acceptance_length=3.0)
    assert "rejection_sample_method=synthetic" in fails(rejection_sample_method="standard", synthetic="chat")
    assert "exceeds max_model_len" in fails(max_model_len=4096, prompt_tokens=4000)
    assert "positions" in fails(k=6, synthetic="chat")
    assert "exactly k=5" in fails(synthetic=[0.5, 0.4])
    assert "non-increasing" in fails(synthetic=[0.1, 0.2, 0.2, 0.1, 0.0])
    assert "[1, 6]" in fails(synthetic_acceptance_length=7.0)
    assert opts(rejection_sample_method="standard")["synthetic"] is None
    assert opts(rejection_sample_method="block")["rejection_sample_method"] == "block"


def test_schedule_validation_mirrors_vllm():
    ok = opts(num_speculative_tokens_per_batch_size=[[9, 32, 2], [1, 8, 5]])
    assert ok["num_speculative_tokens_per_batch_size"] == [[1, 8, 5], [9, 32, 2]]
    for bad in ([], [[2, 8, 5]], [[1, 8, 5], [8, 16, 3]], [[1, 8]], [[1, 8, 9]], [[8, 1, 3]], [[1, 8, 6]], "x",
                [[1, 8, 5.5]]):
        fails(num_speculative_tokens_per_batch_size=bad)
    assert "exceeds k" in fails(k=3, num_speculative_tokens_per_batch_size=[[1, 8, 5]])
    spec = d.speculative_config(ok)
    assert spec["num_speculative_tokens_per_batch_size"] == [(1, 8, 5), (9, 32, 2)]


def test_env_allowlist():
    good = opts(env={"SUFFIX_DSV41_FUSE": "1", "SUFFIX_NVFP4_MOE_TUNE": "1:fused3/32/255/1,5:front4/8/255/1",
                     "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": "2048", "SUFFIX_FP8_DENSE_LAYERS": "*.attn.*"})
    assert good["env"]["SUFFIX_DSV41_FUSE"] == "1"
    assert "allowlist" in fails(env={"SUFFIX_ROCM_QSA_DENSE": "1"})
    assert "no effect at TP=1" in fails(env={"SUFFIX_NCCL_BANDS": "0-inf:har"})
    assert "allowlist" in fails(env={"SUFFIX_MTP_TUNE": "1"})
    assert "allowlist" in fails(env={"PYTHONPATH": "/x"})
    assert "allowlist" in fails(env={"LD_PRELOAD": "/x"})
    assert "set by the tool" in fails(env={"VLLM_ENABLE_V1_MULTIPROCESSING": "1"})
    assert "set by the tool" in fails(env={"VLLM_MOE_ROUTING_SIMULATION_STRATEGY": "uniform_random"})
    assert "characters" in fails(env={"SUFFIX_DSV41_FUSE": "1; rm -rf /"})
    assert "characters" in fails(env={"SUFFIX_DSV41_FUSE": "$(id)"})
    assert "strings" in fails(env={"SUFFIX_DSV41_FUSE": 1})


def test_allowlisted_names_exist_in_the_plugin_or_vllm_naming():
    for name in d.ALLOWED_SUFFIX_ENV:
        assert name.startswith("SUFFIX_") and not name.startswith(("SUFFIX_ROCM", "SUFFIX_NCCL", "SUFFIX_HAR", "SUFFIX_MTP_TUNE"))
    assert all(n.startswith("VLLM_") for n in d.ALLOWED_VLLM_ENV)
    assert not (d.ALLOWED_VLLM_ENV & d.TOOL_OWNED_ENV)


# ---------------------------------------------------------------- synthetic acceptance
def test_synthetic_profiles_are_cumulative_and_match_the_suite():
    for name, rates in d.SYNTHETIC_PROFILES.items():
        assert len(rates) == 5 and all(0 < r <= 1 for r in rates) and list(rates) == sorted(rates, reverse=True), name
    mean = lambda name: 1 + sum(d.SYNTHETIC_PROFILES[name])  # noqa: E731
    # mean accepted length per step vs SPEED.md (chat 2.19-2.32, thinking 3.93-4.02, edit 3.71-3.79)
    assert 2.19 <= mean("chat") <= 2.32 and 3.93 <= mean("thinking") <= 4.02 and 3.6 <= mean("edit") <= 3.8


def test_synthetic_rates_resolution():
    assert d.synthetic_rates(opts(synthetic="thinking", k=3)) == [0.86, 0.71, 0.58]
    assert d.synthetic_rates(opts(synthetic=[0.5, 0.25, 0.1, 0.05, 0.0])) == [0.5, 0.25, 0.1, 0.05, 0.0]
    # length 3.5 -> two certain accepts then half of the third (vLLM's minimum-variance schedule)
    assert d.synthetic_rates(opts(synthetic=None, synthetic_acceptance_length=3.5)) == [1.0, 1.0, 0.5, 0.0, 0.0]
    assert d.synthetic_rates(opts(rejection_sample_method="block")) is None


# ---------------------------------------------------------------- engine kwargs / env / plan
def test_engine_kwargs_mirror_the_prod_flags():
    kw = d.engine_kwargs(opts(), "/m")
    assert kw["model"] == "/m" and kw["load_format"] == "dummy" and kw["skip_tokenizer_init"] is True
    assert kw["tensor_parallel_size"] == 1 and kw["kv_cache_dtype"] == "fp8" and kw["block_size"] == 64
    assert (kw["max_model_len"], kw["max_num_seqs"], kw["max_num_batched_tokens"]) == (524288, 32, 8192)
    assert kw["gpu_memory_utilization"] == 0.93 and kw["enable_prefix_caching"] is True
    assert kw["limit_mm_per_prompt"] == {"image": 0} and kw["disable_log_stats"] is False
    assert "compilation_config" not in kw and "moe_backend" not in kw  # the arch/auto defaults, like prod
    assert kw["speculative_config"] == {"method": "dspark", "num_speculative_tokens": 5,
                                        "rejection_sample_method": "synthetic",
                                        "synthetic_acceptance_rates": [0.61, 0.34, 0.17, 0.09, 0.05]}


def test_engine_kwargs_options():
    kw = d.engine_kwargs(opts(k=3, draft_sample_method="probabilistic", rejection_sample_method="block",
                              enable_adaptive_verification=True, cudagraph_mode="FULL", moe_backend="marlin",
                              async_scheduling=False, enable_flashinfer_autotune=True, kv_cache_dtype="auto",
                              sparse_indexer_topk_backend="torch", linear_backend="cutlass"), "/m")
    assert kw["speculative_config"] == {"method": "dspark", "num_speculative_tokens": 3,
                                        "draft_sample_method": "probabilistic", "rejection_sample_method": "block",
                                        "enable_adaptive_verification": True}
    assert kw["compilation_config"] == {"cudagraph_mode": "FULL"} and kw["moe_backend"] == "marlin"
    assert kw["async_scheduling"] is False and kw["enable_flashinfer_autotune"] is True
    assert "kv_cache_dtype" not in kw and kw["sparse_indexer_topk_backend"] == "torch" and kw["linear_backend"] == "cutlass"
    assert "speculative_config" not in d.engine_kwargs(opts(k=0), "/m")
    assert d.speculative_config(opts(rejection_sample_method="standard")) == {"method": "dspark", "num_speculative_tokens": 5}


def test_env_plan():
    e = d.env_plan(opts())
    assert e["VLLM_USE_V2_MODEL_RUNNER"] == "1" and e["SUFFIX_PRESET"] == "dsv41-sm120"
    assert e["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0" and e["VLLM_LOGGING_STREAM"] == "ext://sys.stderr"
    assert "VLLM_MOE_ROUTING_SIMULATION_STRATEGY" not in e
    assert e["HF_HUB_OFFLINE"] == "1" and e["MALLOC_ARENA_MAX"] == "16"  # d7-spec.json's allocator env
    e = d.env_plan(opts(moe_routing="normal_routing", env={"SUFFIX_PRESET": "dsv41-sm120-comms"}, log_level="DEBUG"))
    assert e["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] == "normal_routing"
    assert e["SUFFIX_PRESET"] == "dsv41-sm120-comms" and e["VLLM_LOGGING_LEVEL"] == "DEBUG"


def test_max_tokens_keeps_every_stream_alive_through_all_windows():
    o = opts(warmup_steps=10, measure_steps=20, profile=True, profile_steps=5)
    assert d.max_tokens_for(o) == (10 + 20 + 5 + 8) * 6 + 1
    assert d.max_tokens_for(opts(k=0, warmup_steps=0, measure_steps=4)) == (4 + 8) * 1 + 1


def test_reexec_only_when_the_env_differs(monkeypatch):
    calls = []
    monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
    monkeypatch.delenv(d._REEXEC_MARK, raising=False)
    d.maybe_reexec({"DSV41_SLICE_TEST_KEY": "1"})
    assert len(calls) == 1 and calls[0][2]["DSV41_SLICE_TEST_KEY"] == "1" and calls[0][2][d._REEXEC_MARK] == "1"
    monkeypatch.setenv(d._REEXEC_MARK, "1")
    d.maybe_reexec({"DSV41_SLICE_TEST_KEY": "2"})  # already re-exec'd: apply in place, never loop
    assert len(calls) == 1 and os.environ["DSV41_SLICE_TEST_KEY"] == "2"
    monkeypatch.delenv(d._REEXEC_MARK)
    d.maybe_reexec({"DSV41_SLICE_TEST_KEY": "2"})  # env already right: nothing to do
    assert len(calls) == 1


# ---------------------------------------------------------------- model dir / tokenizer
def test_write_model_dir_and_files(tmp_path):
    cfg = d.slice_config(opts())[0]
    path, info = d.write_model_dir(str(tmp_path), cfg)
    assert set(os.listdir(path)) == {"config.json", "tokenizer.json", "tokenizer_config.json"}
    assert json.loads((tmp_path / os.path.basename(path) / "config.json").read_text()) == cfg
    assert info["config.json"]["sha256"] == hashlib.sha256((tmp_path / os.path.basename(path) / "config.json").read_bytes()).hexdigest()
    tok = json.loads((tmp_path / os.path.basename(path) / "tokenizer.json").read_text())
    assert len(tok["model"]["vocab"]) == 16160 and tok["model"]["vocab"]["t16159"] == 16159


def test_tokenizer_loads_and_has_the_compressed_vocab_the_config_declares(tmp_path):
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from tokenizers import Regex, normalizers  # noqa: F401

    cfg = d.slice_config(opts())[0]
    path, _ = d.write_model_dir(str(tmp_path), cfg)
    tok = transformers.AutoTokenizer.from_pretrained(path)
    assert len(tok) == 16160
    # vllm.models.deepseek_v41.common.engram.build_compressed_token_map, copied (Apache-2.0)
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " ")])
    backend = tok.backend_tokenizer
    keys = {}
    for token_id in range(len(tok)):
        text = backend.decode([token_id], skip_special_tokens=False)
        key = backend.id_to_token(token_id) if "�" in text else (normalizer.normalize_str(text) or text)
        keys.setdefault(key, len(keys))
    assert len(keys) == cfg["text_config"]["engram_compressed_vocab_size"] == 16160


# ---------------------------------------------------------------- CLI
def run_cli(argv):
    out = io.StringIO()
    rc = d.main(argv, stdout=out)
    return rc, out.getvalue()


def test_dry_cli_prints_the_plan_and_cleans_up(tmp_path):
    rc, text = run_cli(["--dry", "--opts", json.dumps({"conc": [1, 8], "k": 3, "work_dir": str(tmp_path)})])
    assert rc == 0 and text.count("\n") == 1
    doc = json.loads(text)
    assert doc["schema"] == "dsv41-slice/1" and doc["dry"] is True and doc["ok"] is True
    plan = doc["plan"]
    assert plan["engine_kwargs"]["speculative_config"]["num_speculative_tokens"] == 3
    assert set(plan["model_dir_files"]) == {"config.json", "tokenizer.json", "tokenizer_config.json"}
    assert plan["config_diff"]["text_config.num_attention_heads"] == [64, 8]
    assert list(tmp_path.iterdir()) == []  # the temp model dir is gone
    assert plan["live_reference_ms"] == {"1": 12.2, "8": 25.9}
    assert plan["step_stream_gb"]["1"]["prod_rank"] < plan["step_stream_gb"]["8"]["prod_rank"]


def test_dry_cli_keep_dir_and_out_file(tmp_path):
    out = tmp_path / "plan.json"
    rc, text = run_cli(["--dry", "--keep-dir", "--out", str(out), "--opts", json.dumps({"work_dir": str(tmp_path)})])
    assert rc == 0 and json.loads(out.read_text()) == json.loads(text)
    kept = json.loads(text)["model_dir"]
    assert os.path.isfile(os.path.join(kept, "config.json"))


def test_cli_exit_codes_and_messages(capsys):
    assert d.main(["--opts", "{not json"]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert d.main(["--dry", "--opts", '{"k": 99}']) == 2
    assert "opts:" in capsys.readouterr().err
    assert d.main(["--dry", "--opts", '{"nope": 1}']) == 2
    assert d.main(["--dry", "--opts", "[1]"]) == 2
    assert d.main(["--opts-file", "/nonexistent/opts.json", "--dry"]) == 2
    capsys.readouterr()


def test_print_schema_lists_every_opt_and_the_env_allowlist():
    rc, text = run_cli(["--print-schema"])
    doc = json.loads(text)
    assert rc == 0 and set(doc["opts"]) == set(d.SCHEMA) and "conc" in doc["opts"]
    assert doc["opts"]["gpu_memory_utilization"]["min"] == 0.3 and doc["opts"]["block_size"]["enum"] == [32, 64]
    assert "SUFFIX_DSV41_FUSE" in doc["allowed_env"]["SUFFIX"] and "memcpy" in doc["families"]


# ---------------------------------------------------------------- pure measurement helpers
def test_step_stats():
    s = d.step_stats([0.010, 0.012, 0.011, 0.030], [12, 12, 13, 12], 0.063, 2)
    assert s["steps"] == 4 and s["tokens_emitted"] == 49
    assert s["step_ms"]["mean"] == pytest.approx(15.75) and s["step_ms"]["p50"] == pytest.approx(11.5)
    assert s["step_ms"]["max"] == pytest.approx(30.0) and s["step_ms"]["min"] == pytest.approx(10.0)
    assert s["tokens_per_stream_per_step"] == pytest.approx(49 / 8)
    assert s["tok_s_per_stream"] == pytest.approx(49 / 2 / 0.063)


def test_spec_delta():
    before = {"drafts": 100, "draft_tokens": 500, "accepted": 100, "per_pos": [50, 25, 15, 7, 3]}
    after = {"drafts": 200, "draft_tokens": 1000, "accepted": 300, "per_pos": [110, 75, 55, 40, 20]}
    s = d.spec_delta(before, after, 8)
    assert s["drafts"] == 100 and s["accepted_len"] == pytest.approx(3.0) and s["verified_per_stream"] == pytest.approx(6.0)
    assert s["verified_tokens_per_step"] == pytest.approx(48.0) and s["per_pos_cum_accept"] == [0.6, 0.5, 0.4, 0.33, 0.17]
    assert s["acceptance_rate"] == pytest.approx(0.4)
    assert d.spec_delta(None, after, 8) is None
    assert d.spec_delta(before, before, 8)["drafts"] == 0


@pytest.mark.parametrize("name,family", [
    ("Memcpy HtoD (Pinned -> Device)", "memcpy"), ("Memset (Device)", "memcpy"),
    ("_engram_lookup_kernel", "engram"), ("_hash_ids_kernel", "engram"),
    ("mhc_post_tilelang_kernel", "mhc"), ("hc_collapse_kernel", "mhc"),
    ("void deep_gemm::sm90_fp8_paged_mqa_logits", "indexer"), ("top_k_per_row_decode", "indexer"),
    ("compress_quant_cache_kernel", "compressor"), ("markov_bias_add", "draft"),
    ("gumbel_sample_kernel", "sampler"), ("rejection_sample_kernel", "sampler"),
    ("void flashinfer::BatchMLAPagedAttentionKernel", "attention"),
    ("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert", "attention"),
    ("cutlass_fused_moe_nvfp4", "moe"), ("void cutlass::device_kernel<GemmGrouped>", "moe"),
    ("fused_topk_bias_router", "moe"), ("rms_norm_kernel", "norm_quant"),
    ("quantize_with_block_size", "norm_quant"), ("nvjet_sm120_tst_128x64", "dense_gemm"),
    ("void at::native::vectorized_elementwise_kernel", "elementwise"), ("triton_poi_fused_0", "other"),
])
def test_kernel_classifier(name, family):
    assert d.classify_kernel(name) == family


def test_merge_span_and_aggregate():
    assert d.merge_span_us([(0, 10), (5, 15), (20, 30)]) == 25
    assert d.merge_span_us([(0, 10), (2, 3)]) == 10 and d.merge_span_us([]) == 0
    events = [("cutlass_fused_moe", 0, 40), ("cutlass_fused_moe", 100, 140), ("rms_norm_kernel", 40, 45),
              ("rms_norm_kernel", 140, 145), ("mystery", 10, 20), ("flashinfer::BatchMLAPagedAttention", 20, 30)]
    agg = d.aggregate_kernels(events, steps=2, wall_ms_per_step=0.1)
    assert agg["kernel_us_per_step"] == pytest.approx((80 + 10 + 10 + 10) / 2)
    assert agg["span_us_per_step"] == pytest.approx((45 + 45) / 2)
    fams = {f["family"]: f for f in agg["families"]}
    assert fams["moe"]["us_per_step"] == 40.0 and fams["moe"]["calls_per_step"] == 1.0
    assert agg["families"][0]["family"] == "moe" and sum(f["share"] for f in agg["families"]) == pytest.approx(1.0, abs=1e-3)
    assert agg["top_kernels"][0]["name"] == "cutlass_fused_moe" and agg["unclassified_top"][0]["name"] == "mystery"
    assert agg["gpu_busy_fraction"] == pytest.approx(45 / 100)
    assert d.aggregate_kernels([], steps=3)["kernel_us_per_step"] == 0


PROD_LOG = """\
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:14:10 [attention.py:170] Using DeepSeek's fp8_ds_mla KV cache format.
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:14:09 [__init__.py:993] Using FlashInferCutlassMxfp8LinearKernel for MXFP8 GEMM
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:14:10 [nvfp4.py:314] Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends: ['FLASHINFER_TRTLLM']
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:14:16 [engram.py.dsv41-engram-mxfp4.py:1056] Engram table offloaded to pinned host memory: 48000217 rows x 256, 6.08 GiB per rank
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:14:16 [attention.py:1368] Using FP8 indexer cache for Lightning Indexer.
[suffix dsv41-engram-mxfp4] ACTIVE: vllm.models.deepseek_v41.common.engram replaced by vllm#61530 (MXFP4 tables).
[suffix dsv41-sm120-block64] ACTIVE: DeepseekV41IndexerBackend declares 64-token kernel blocks on SM12x (vllm#60762).
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:14:09 [gpu_worker.py:510] Using V2 Model Runner
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:18:27 [dspark.py:487] DSpark draft model loaded: 97 params
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:18:38 [mxfp4.py:818] Using 'DEEPGEMM_MXFP4' Mxfp4 MoE backend.
(Worker_TP6_EP6 pid=1378) INFO 10-17 17:18:32 [model_runner.py:429] Model loading took 40.66 GiB memory and 262.634288 seconds
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:18:33 [breakable_cudagraph.py:294] Breakable CUDA graph enabled
(Worker_TP1_EP1 pid=1373) INFO 10-17 17:21:33 [model_runner.py:1140] Graph capturing finished in 89 secs, took 2.69 GiB
(Worker_TP0_EP0 pid=1372) INFO 10-17 17:21:33 [gpu_worker.py:763] Available KV cache memory: 37.17 GiB
(EngineCore pid=1159) INFO 10-17 17:21:34 [kv_cache_utils.py:1101] GPU KV cache size: 10,867,561 tokens, Maximum concurrency for 262,144 tokens per request: 41.46x
"""


def test_engine_checks_parse_the_real_prod_boot_log():
    got = d.evaluate_checks(PROD_LOG)
    assert all(c["found"] is not None for c in got.values()), {k: v for k, v in got.items() if v["found"] is None}
    assert all(c["match"] is not False for c in got.values())
    assert got["nvfp4_moe_backend"]["found"] == "FLASHINFER_CUTLASS" and got["model_load_gib"]["found"] == "40.66"
    assert got["kv_cache_tokens"]["found"] == "10,867,561" and got["graph_pool_gib"]["found"] == "2.69"
    bad = d.evaluate_checks(PROD_LOG.replace("FLASHINFER_CUTLASS", "MARLIN"))
    assert bad["nvfp4_moe_backend"]["match"] is False
    missing = d.evaluate_checks("nothing here")
    assert missing["engram_mxfp4_overlay"]["found"] is None and missing["engram_mxfp4_overlay"]["match"] is False


def test_recorder_emits_once_and_cleans_numbers():
    buf = io.StringIO()
    rec = d.Recorder(buf)
    rec.doc["results"]["8"] = {"x": float("nan"), "y": 1.23456789012, "z": [float("inf"), 2]}
    rec.warn("w")
    rec.emit(4, partial=True)
    rec.emit(0)  # a second emit (deadline thread vs main thread) must not print again
    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    doc = json.loads(lines[0])
    assert doc["exit"] == 4 and doc["ok"] is False and doc["partial"] is True and doc["warnings"] == ["w"]
    assert doc["results"]["8"] == {"x": None, "y": 1.23457, "z": [None, 2]}


def test_clock_raises_with_the_deadline_code():
    clock = d.Clock(0)
    with pytest.raises(d.SliceError) as ei:
        while True:
            clock.check()
    assert ei.value.code == 4


# ---------------------------------------------------------------- orchestration against a fake vLLM
import sys  # noqa: E402
import types  # noqa: E402


class _Out:
    def __init__(self, rid, ntok, finished=False):
        self.request_id, self.finished = rid, finished
        self.outputs = [types.SimpleNamespace(token_ids=list(range(ntok)))]


class FakeEngine:
    """The slice of LLMEngine the tool uses: add_request / step / abort_request / has_unfinished_requests."""

    def __init__(self, k=5, per_step=3, finish_after=None):
        self.k, self.per_step, self.finish_after = k, per_step, finish_after
        self.reqs, self.calls, self.params = {}, 0, None
        self.drafts = self.draft_tokens = self.accepted = 0
        self.per_pos = [0] * k
        self.vllm_config = types.SimpleNamespace(cache_config=types.SimpleNamespace(num_gpu_blocks=1234))

    def add_request(self, rid, prompt, params):
        assert isinstance(prompt["prompt_token_ids"], list) and params.kw["detokenize"] is False
        self.reqs[rid], self.params = 0, params

    def step(self):
        self.calls += 1
        if self.calls == 1:  # async pipeline fill: nothing to return yet
            return []
        outs = []
        for rid in list(self.reqs):
            first = self.reqs[rid] == 0
            n = 1 if first else self.per_step
            self.reqs[rid] += n
            if not first:
                self.drafts += 1
                self.draft_tokens += self.k
                self.accepted += n - 1
                for i in range(n - 1):
                    self.per_pos[i] += 1
            done = self.finish_after is not None and self.reqs[rid] >= self.finish_after
            outs.append(_Out(rid, n, finished=done))
        return outs

    def abort_request(self, ids):
        for i in ids:
            self.reqs.pop(i, None)

    def has_unfinished_requests(self):
        return bool(self.reqs)


def install_fake_vllm(monkeypatch, engine=None, boom=None):
    engine = engine or FakeEngine()
    ns = types.SimpleNamespace

    class FakeLLM:
        last_kwargs = None

        def __init__(self, **kw):
            if boom:
                raise boom
            FakeLLM.last_kwargs = kw
            self.llm_engine = engine

        def get_metrics(self):
            return [ns(name="vllm:spec_decode_num_drafts", value=engine.drafts),
                    ns(name="vllm:spec_decode_num_draft_tokens", value=engine.draft_tokens),
                    ns(name="vllm:spec_decode_num_accepted_tokens", value=engine.accepted),
                    ns(name="vllm:spec_decode_num_accepted_tokens_per_pos", values=list(engine.per_pos)),
                    ns(name="vllm:other", value=1)]

    class FakeSamplingParams:
        def __init__(self, **kw):
            self.kw = kw

    mod = types.ModuleType("vllm")
    mod.LLM, mod.SamplingParams, mod.__version__ = FakeLLM, FakeSamplingParams, d.VLLM_VERSION_PREFIX + "+gfake"
    sp = types.ModuleType("vllm.sampling_params")
    sp.RequestOutputKind = ns(DELTA="delta")
    monkeypatch.setitem(sys.modules, "vllm", mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sp)
    monkeypatch.setattr(d, "collect_versions", lambda: {"torch": "x", "vllm": mod.__version__, "cuda": "13.0",
                                                       "cuda_available": True})
    monkeypatch.setattr(d, "gpu_properties", lambda: {"name": "fake", "cc": "12.0"})
    monkeypatch.setattr(d, "apply_shared_expert_patch", lambda *a, **k: None)  # needs the real vLLM source
    return engine, FakeLLM


def run_fake(o, rec=None, tee_text=PROD_LOG):
    pytest.importorskip("torch")
    rec = rec or d.Recorder(io.StringIO())
    plan = d.build_plan(o, "/m")
    plan["slice_vocab"] = 16160
    tee = types.SimpleNamespace(sync=lambda *a, **k: None, text=lambda: tee_text)
    code = d.run_slice(o, rec, tee, d.Clock(60), "/m", plan)
    return code, rec


def test_run_slice_measures_every_concurrency(monkeypatch):
    engine, llm_cls = install_fake_vllm(monkeypatch)
    o = opts(conc=[2, 4], warmup_steps=3, measure_steps=10, shared_expert_tp=False)
    code, rec = run_fake(o)
    assert code == 0
    assert llm_cls.last_kwargs["speculative_config"]["num_speculative_tokens"] == 5
    res = rec.doc["results"]
    assert set(res) == {"2", "4"} and engine.reqs == {}  # every stream aborted and drained
    r = res["2"]
    assert r["steps"] == 10 and r["tokens_emitted"] == 10 * 2 * 3 and r["tokens_per_stream_per_step"] == 3
    assert r["step_ms"]["mean"] == pytest.approx(r["wall_s"] / 10 * 1e3) and r["tok_s_per_stream"] > 0
    assert r["spec"]["accepted_len"] == pytest.approx(3.0) and r["spec"]["verified_per_stream"] == pytest.approx(6.0)
    assert r["spec"]["verified_tokens_per_step"] == pytest.approx(12.0) and r["spec"]["drafts"] == 20
    assert r["spec"]["per_pos_cum_accept"] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert r["slice_step_ms"] == r["step_ms"]["mean"] and r["live_step_ms"] is None and r["expected_slice_ms"] is None
    assert r["prod_equivalent_ms_est"] == pytest.approx(r["slice_step_ms"] - d.bias_terms(o, 2)["quantified_total"], abs=1e-3)
    assert r["synthetic_check"]["max_abs_z"] > 4.5  # fake engine accepts 2 of 5 always: far from the chat profile
    assert any("deviates" in w for w in rec.doc["warnings"])
    assert engine.params.kw["max_tokens"] == d.max_tokens_for(o) and engine.params.kw["ignore_eos"] is True
    assert rec.doc["engine"]["num_gpu_blocks"] == 1234
    assert rec.doc["engine"]["log_checks"]["nvfp4_moe_backend"]["match"] is True
    assert set(rec.doc["phases"]) >= {"import_vllm", "engine_init", "measure_c2", "measure_c4"}
    assert rec.doc["versions"]["gpu"]["name"] == "fake"


def test_run_slice_reports_the_live_reference_and_k0(monkeypatch):
    install_fake_vllm(monkeypatch, FakeEngine(k=1, per_step=1))
    code, rec = run_fake(opts(conc=[1], k=0, warmup_steps=0, measure_steps=4))
    assert code == 0 and rec.doc["results"]["1"]["live_step_ms"] == 12.2 and rec.doc["results"]["1"]["expected_slice_ms"] is None
    assert rec.doc["results"]["1"]["tokens_per_stream_per_step"] == 1


def test_live_concurrency_gets_the_calibration_target(monkeypatch):
    install_fake_vllm(monkeypatch)
    code, rec = run_fake(opts(conc=[8], warmup_steps=0, measure_steps=4))
    r = rec.doc["results"]["8"]
    assert r["expected_slice_ms"] == pytest.approx(21.56, abs=0.05)
    assert r["ratio_to_expected"] == pytest.approx(r["slice_step_ms"] / r["expected_slice_ms"], abs=1e-3)


def test_engine_refusal_keeps_the_verbatim_text(monkeypatch):
    install_fake_vllm(monkeypatch, boom=ValueError("Adaptive verification trims verification requests on device, "
                                                   "which the FLASHINFER backend does not support."))
    with pytest.raises(d.SliceError) as ei:
        run_fake(opts(enable_adaptive_verification=True))
    assert ei.value.code == 3 and "Adaptive verification trims" in str(ei.value)


def test_engine_error_is_recorded_in_the_json(monkeypatch):
    install_fake_vllm(monkeypatch, boom=RuntimeError("block size 64 is not supported"))
    rec = d.Recorder(io.StringIO())
    with pytest.raises(d.SliceError):
        run_fake(opts(), rec)
    assert rec.doc["engine_error"] == "RuntimeError: block size 64 is not supported"
    assert "Traceback" in rec.doc["engine_traceback"] and "RuntimeError" in rec.doc["engine_traceback"]


def test_streams_finishing_inside_the_window_are_flagged(monkeypatch):
    install_fake_vllm(monkeypatch, FakeEngine(finish_after=10))
    code, rec = run_fake(opts(conc=[2], warmup_steps=0, measure_steps=8))
    assert any("finished inside the window" in w for w in rec.doc["warnings"])


def test_deadline_stops_the_loop(monkeypatch):
    install_fake_vllm(monkeypatch)
    pytest.importorskip("torch")
    o = opts(conc=[2])
    plan = d.build_plan(o, "/m")
    plan["slice_vocab"] = 16160
    tee = types.SimpleNamespace(sync=lambda *a, **k: None, text=lambda: "")
    with pytest.raises(d.SliceError) as ei:
        d.run_slice(o, d.Recorder(io.StringIO()), tee, d.Clock(-1), "/m", plan)
    assert ei.value.code == 4


def test_engine_check_mismatch_becomes_a_warning(monkeypatch):
    install_fake_vllm(monkeypatch)
    code, rec = run_fake(opts(conc=[1], warmup_steps=0, measure_steps=4), tee_text=PROD_LOG.replace("FLASHINFER_CUTLASS", "MARLIN"))
    assert any("nvfp4_moe_backend" in w and "MARLIN" in w for w in rec.doc["warnings"])


def test_shared_expert_patch_refuses_on_source_drift(monkeypatch):
    # a vLLM whose DeepseekV4MoE.__init__ no longer builds the shared expert the pinned way
    mod = types.ModuleType("vllm.models.deepseek_v4.nvidia.model")

    class MoE:
        def __init__(self):
            self.shared = None  # no anchor lines here

    mod.DeepseekV4MoE, mod.DeepseekV4MLP = MoE, object
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v4.nvidia.model", mod)
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.nvidia"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    with pytest.raises(d.SliceError) as ei:
        d.apply_shared_expert_patch()
    assert ei.value.code == 6 and "shared_expert_tp=false" in str(ei.value)


def test_shared_expert_patch_shards_only_the_shared_experts(monkeypatch):
    src = d._SHARED_ANCHORS
    built = []

    class MLP:
        def __init__(self, hidden_size, intermediate_size, prefix=""):
            built.append((prefix, intermediate_size))

    class MoE:
        def __init__(self):  # carries the three anchor strings, like the pinned source
            intermediate_size = 2304  # intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(hidden_size=5120, intermediate_size=intermediate_size, prefix="layers.0.ffn.shared_experts")

    mod = types.ModuleType("vllm.models.deepseek_v4.nvidia.model")
    mod.DeepseekV4MoE, mod.DeepseekV4MLP = MoE, MLP
    monkeypatch.setitem(sys.modules, "vllm.models.deepseek_v4.nvidia.model", mod)
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v4", "vllm.models.deepseek_v4.nvidia"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(d, "_SHARED_ANCHORS", ("intermediate_size = 2304", "self.shared_experts = MLP(", 'prefix="layers.0.ffn.shared_experts"'))
    d.apply_shared_expert_patch()
    mod.DeepseekV4MLP(hidden_size=5120, intermediate_size=2304, prefix="layers.3.ffn.shared_experts")
    mod.DeepseekV4MLP(hidden_size=5120, intermediate_size=2304, prefix="layers.3.ffn.experts.0")
    assert built == [("layers.3.ffn.shared_experts", 288), ("layers.3.ffn.experts.0", 2304)]
    assert src  # the real anchors are pinned strings
