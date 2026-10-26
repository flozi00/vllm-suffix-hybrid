# SPDX-License-Identifier: Apache-2.0
"""suffix_hybrid/tools/helion_tune.py on CPU: per-rank model dicts of a Qwen4Exp-style config
(text_config), every family's cases() on them, the winner merge, --cases-only, the parent/child
plumbing and the sitecustomize SUFFIX_HELION_TUNE gate. Silicon: the in-pod run itself."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from suffix_hybrid import helion_kernels as hk
from suffix_hybrid.tools import helion_tune as ht

ROOT = Path(__file__).resolve().parents[1]
# Qwen3.8-Flash-Next-like (Qwen4ExpForConditionalGeneration + Qwen4ExpMTP): dims of
# kernels/nvfp4_gemm.QWEN_FLASH_SHAPES / kernels/fp8_moe.py, ModelOpt mixed precision.
CONFIG = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.language_model.layers.0.mlp.experts.0.gate_proj": {"quant_algo": "NVFP4"},
            "mtp.layers.0.mlp.experts.0.gate_proj": {"quant_algo": "FP8_BLOCK"}}},
    "text_config": {
        "hidden_size": 2560, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
        "intermediate_size": 5120, "moe_intermediate_size": 640, "num_experts": 512,
        "num_experts_per_tok": 10, "shared_expert_intermediate_size": 640, "vocab_size": 248320,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "mtp_num_hidden_layers": 1, "mlp_only_layers": []},
}
NAME = "qwen3.8-flash-next-nvfp4"


def _env():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    env["PYTHONPATH"] = str(ROOT)
    return env


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("ep", [False, True])
def test_model_dict_per_rank(tp, ep):
    m = ht.model_dict(CONFIG, tp, ep, NAME)
    ep_on = ep and tp > 1
    assert (m["architecture"], m["tp"], m["ep"], m["quant_format"]) == (
        "Qwen4ExpForConditionalGeneration", tp, tp if ep_on else 1, "nvfp4")
    assert (m["hidden"], m["topk"], m["head_dim"], m["experts_global"]) == (2560, 10, 256, 512)
    assert m["experts"] == (512 // tp if ep_on else 512)
    assert m["moe_intermediate"] == (640 if ep_on else 640 // tp)
    assert (m["heads"], m["kv_heads"], m["vocab"]) == (24 // tp, max(1, 2 // tp), 248320 // tp)
    assert m["intermediate"] is None  # every layer is MoE: no dense MLP linears
    # GDN dims stay whole-model (kgdn1.cases divides by tp itself)
    assert [m[k] for k in ("linear_num_key_heads", "linear_num_value_heads",
                           "linear_key_head_dim", "linear_value_head_dim")] == [16, 48, 128, 128]
    lin = {x["name"]: (x["n"], x["k"]) for x in m["linears"]}
    assert lin["gdn_in_proj_qkvz"] == (16384 // tp, 2560)
    assert lin["gdn_in_proj_ba"] == (96 // tp, 2560)
    assert lin["gdn_out_proj"] == lin["o_proj"] == (2560, 6144 // tp)
    assert lin["qkv_proj"] == ((48 // tp + 2 * max(1, 2 // tp)) * 256, 2560)  # gated q + k + v
    assert lin["shared_gate_up"] == (1280 // tp, 2560)
    assert lin["shared_down"] == (2560, 640 // tp)
    assert lin["lm_head"] == (248320 // tp, 2560)


def test_tp2_linears_match_the_qwen_flash_table():
    from suffix_hybrid.kernels.nvfp4_gemm import QWEN_FLASH_SHAPES as T
    lin = {x["name"]: (x["n"], x["k"]) for x in ht.model_dict(CONFIG, 2, False)["linears"]}
    assert lin["gdn_in_proj_qkvz"] == T["flash_gdn_in_proj_qkvz"]
    assert lin["gdn_in_proj_ba"] == T["flash_gdn_in_proj_ba"]
    assert lin["o_proj"] == lin["gdn_out_proj"] == T["flash_o_proj"]
    assert lin["qkv_proj"] == T["flash_qsa_qkv"]
    assert lin["shared_gate_up"] == T["flash_shared_gate_up"]
    assert lin["shared_down"] == T["flash_shared_down"]


def test_layouts_ep_modes_and_mtp():
    labels = lambda mode: [lab for lab, _ in ht.layouts(CONFIG, [1, 2], mode, NAME)]
    assert labels("auto") == ["tp1", "tp1-mtp", "tp2", "tp2-mtp", "tp2-ep", "tp2-ep-mtp"]
    assert labels("on") == ["tp1", "tp1-mtp", "tp2-ep", "tp2-ep-mtp"]
    assert labels("off") == ["tp1", "tp1-mtp", "tp2", "tp2-mtp"]
    mtp = dict(ht.layouts(CONFIG, [2], "on", NAME))["tp2-ep-mtp"]
    assert (mtp["architecture"], mtp["quant_format"], mtp["block"], mtp["experts"]) == (
        "Qwen4ExpMTP", "fp8_block", [128, 128], 256)
    assert mtp["linears"] == [] and mtp["linear_num_key_heads"] is None
    assert [lab for lab, _ in ht.layouts(CONFIG, [2], "on", NAME, "off")] == ["tp2-ep"]


def test_mtp_quant_detection():
    assert ht.mtp_quant({"quantized_layers": {"mtp.fc": {"quant_algo": "FP8"}}}, "nvfp4") == "fp8_block"
    assert ht.mtp_quant({"ignore": ["lm_head", "mtp.*"]}, "nvfp4") is None
    assert ht.mtp_quant({"quant_algo": "NVFP4"}, "nvfp4") == "nvfp4"
    assert ht.quant_format({"quant_algo": "FP8", "weight_block_size": [128, 128]}) == "fp8_block"
    assert ht.quant_format({}) == "bf16"


def test_indivisible_layout_is_reported_not_raised(capsys):
    cfg = json.loads(json.dumps(CONFIG))
    cfg["text_config"]["num_attention_heads"] = 20
    assert [lab for lab, _ in ht.layouts(cfg, [8], "off")] == []
    assert "not divisible" in capsys.readouterr().out


@pytest.mark.parametrize("family", ht.FAMILIES)
def test_every_family_accepts_the_model_dicts(family):
    import importlib
    mod = importlib.import_module(f"suffix_hybrid.helion_kernels.{family}")
    models = ht.layouts(CONFIG, [1, 2, 4, 8], "auto", NAME)
    for kernel in mod.KERNELS:
        for _, m in models:
            for case in kernel["cases"](m):
                assert isinstance(case, dict) and isinstance(case["numtokens"], int)
        got = ht.collect(kernel, models)
        assert [c["numtokens"] for c, _ in got] == sorted(c["numtokens"] for c, _ in got)
        assert len({hk.case_key(c) for c, _ in got}) == len(got)


def test_nvfp4_moe_ep_cases_key_like_the_serving_lane():
    from suffix_hybrid.helion_kernels import nvfp4_moe as hm
    models = ht.layouts(CONFIG, [1, 2, 4, 8], "auto", NAME)
    kernel = next(k for k in hm.KERNELS if k["name"] == hm.FC1)
    got = ht.collect(kernel, models)
    local = sorted({c["experts"] for c, _ in got})
    assert local == [64, 128, 256]  # EP ranks only: E 512 is past the oxide baseline's 256
    for case, labels in got:
        assert all(lab.endswith("-ep") for lab in labels)
        assert case == hm.lane_case(case["experts"], 2560, 640, 10, 0, case["numtokens"], None, 512)
    assert hm.ep_base(got[0][0]) > 0  # the lab routes a middle EP rank's id window


def test_fp8_moe_tunes_only_the_fp8_mtp_ep_ranks():
    from suffix_hybrid.helion_kernels import fp8_moe as hf
    got = ht.collect(hf.KERNELS[0], ht.layouts(CONFIG, [1, 2, 4, 8], "auto", NAME))
    assert got and all(lab.endswith("-ep-mtp") for _, labels in got for lab in labels)
    assert sorted({c["experts"] for c, _ in got}) == [64, 128, 256]


def test_nvfp4_gemm_uses_the_arch_table_at_tp2():
    from suffix_hybrid.helion_kernels import nvfp4_gemm as hg
    from suffix_hybrid.kernels.nvfp4_gemm import QWEN_FLASH_SHAPES
    m = ht.model_dict(CONFIG, 2, True, NAME)
    assert set(QWEN_FLASH_SHAPES.values()) <= set(hg.dense_shapes(m))


def test_collect_filters():
    from suffix_hybrid.helion_kernels import nvfp4_gemm as hg
    models = ht.layouts(CONFIG, [2], "off", NAME)
    assert ht.collect(hg.KERNELS[0], models, kernels=["nope"]) == []
    assert {c["numtokens"] for c, _ in ht.collect(hg.KERNELS[0], models, numtokens=[1, 8])} == {1, 8}


def _rec(case, us, speedup, shipped_us=None, status="ok", correct=True):
    return {"case": case, "config": {"num_warps": int(us)}, "helion_us": us, "speedup": speedup,
            "shipped_us": shipped_us, "status": status, "correct": correct}


def test_merge_ships_only_wins_and_replaces_only_when_faster():
    a, b, c, d, e = ({"numtokens": n, "h": 16} for n in (1, 8, 32, 64, 128))
    shipped = [{"key": a, "config": {"old": 1}}, {"key": hk.case_key(b), "config": {"old": 2}}]
    recs = [_rec(a, 5.0, 1.3, shipped_us=4.0),      # faster than oxide, slower than shipped: keep
            _rec(b, 3.0, 1.2, shipped_us=4.0),      # faster than shipped: replace
            _rec(c, 2.0, 0.9),                      # loses to the baseline: never shipped
            _rec(d, 2.0, 1.5, correct=False, status="incorrect"),
            _rec(e, 9.0, 1.1), _rec(e, 8.0, 1.2)]   # new: the faster of two records
    rows, added, replaced = ht.merge(shipped, recs)
    assert (added, replaced) == (1, 1)
    assert [hk.case_key(r["key"]) if isinstance(r["key"], dict) else r["key"] for r in rows] == \
        sorted(hk.case_key(x) for x in (a, b, e))
    by = {hk.case_key(r["key"]) if isinstance(r["key"], dict) else r["key"]: r["config"] for r in rows}
    assert by[hk.case_key(a)] == {"old": 1}
    assert by[hk.case_key(b)] == {"num_warps": 3}
    assert by[hk.case_key(e)] == {"num_warps": 8}
    # a shipped config that could not run (shipped_us None) is replaced by a win
    rows, _, replaced = ht.merge(shipped, [_rec(a, 5.0, 1.3)])
    assert replaced == 1 and rows[0]["config"] == {"num_warps": 5}
    # the loader reads what merge writes
    assert all(set(r) == {"key", "config"} for r in rows)
    assert ht.merge(shipped, [_rec(a, 5.0, 1.3)], min_speedup=1.5)[1:] == (0, 0)


def test_load_config_forms(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    assert ht.load_config(str(tmp_path)) == CONFIG
    assert ht.load_config(str(tmp_path / "config.json")) == CONFIG
    assert ht.load_config(json.dumps(CONFIG)) == CONFIG
    assert ht.model_name("s3://model-weights/qwen3.8-flash-next-nvfp4/config.json", {}) == NAME
    assert ht.model_name("s3://model-weights/qwen3.8-flash-next-nvfp4/", {}) == NAME
    assert ht.model_name("{}", {"_name_or_path": "org/" + NAME}) == NAME


def _tool(*args, timeout=120):
    return subprocess.run([sys.executable, "-m", "suffix_hybrid.tools.helion_tune", *args],
                          capture_output=True, text=True, env=_env(), cwd=str(ROOT), timeout=timeout)


def _recs(stdout):
    tag = f"{ht.PREFIX} REC "
    return [json.loads(x[len(tag):]) for x in stdout.splitlines() if x.startswith(tag)]


def test_cases_only_lists_without_a_gpu():
    out = _tool("--config", json.dumps(CONFIG), "--name", NAME, "--tp", "2", "--ep", "on",
                "--families", "nvfp4_moe,fp8_moe", "--numtokens", "1,8", "--cases-only")
    assert out.returncode == 0, out.stderr
    recs = _recs(out.stdout)
    assert [r["layout"] for r in recs if r["kind"] == "model"] == ["tp2-ep", "tp2-ep-mtp"]
    counts = {r["kernel"]: r["cases"] for r in recs if r["kind"] == "count"}
    assert counts["nvfp4_moe_fc1"] == 2 and counts["fp8_moe_route_quant"] == 2
    cases = [r for r in recs if r["kind"] == "case"]
    assert len(cases) == sum(counts.values())
    assert out.stdout.rstrip().endswith(f"{ht.PREFIX} DONE")
    assert f"{ht.PREFIX} CONFIG" not in out.stdout


def test_family_child_failure_is_a_record_not_a_crash():
    # CPU host: the child cannot read the GPU -> one error record; the parent still finishes.
    out = _tool("--config", json.dumps(CONFIG), "--tp", "1", "--families", "kgdn1")
    assert out.returncode == 0, out.stderr
    errs = [r for r in _recs(out.stdout) if r["kind"] == "record" and r.get("status") == "error"]
    assert errs and errs[0]["family"] == "kgdn1"
    assert out.stdout.rstrip().endswith(f"{ht.PREFIX} DONE")


def test_unknown_family_refused():
    out = _tool("--config", "{}", "--families", "nope", "--cases-only")
    assert out.returncode == 2 and "unknown families" in out.stderr


def test_sitecustomize_gate_runs_once_then_holds(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CONFIG))
    env = {**_env(), "SUFFIX_HELION_TUNE": f"--config {cfg} --tp 1 --families nvfp4_gemm --cases-only"}
    # a child of the tuning tree (marker inherited): the gate is inert
    quiet = subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True,
                           env={**env, "SUFFIX_HELION_TUNE_ACTIVE": "1"}, timeout=60)
    assert quiet.stdout == "ok\n" and "helion-tune" not in quiet.stderr
    # unset: no behaviour change
    plain = subprocess.run([sys.executable, "-c", "print('ok')"], capture_output=True, text=True,
                           env=_env(), timeout=60)
    assert plain.stdout == "ok\n"
    # top-level process: the tool runs (its own python does not re-enter the gate), then holds
    proc = subprocess.Popen([sys.executable, "-c", "print('vllm would start')"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    lines, deadline = [], time.time() + 120
    try:
        for line in proc.stdout:
            lines.append(line)
            if "holding the pod" in line or time.time() > deadline:
                break
        assert proc.poll() is None  # still held
    finally:
        proc.kill()
        proc.wait()
    text = "".join(lines)
    assert text.count(f"{ht.PREFIX} DONE") == 1
    assert "exit 0; holding the pod" in text and "vllm would start" not in text


def test_spec_k_reaches_kgdn1_verify_cases(capsys):
    from suffix_hybrid.tools import helion_tune as ht
    cfg = json.dumps(CONFIG)
    assert ht.main(["--config", cfg, "--tp", "2", "--families", "kgdn1", "--cases-only",
                    "--spec-k", "4"]) in (0, None)
    out = capsys.readouterr().out
    verify = [l for l in out.splitlines() if '"kernel": "kgdn1_verify"' in l and '"kind": "count"' in l]
    assert verify and '"cases": 0' not in verify[0], out[-2000:]


def test_gpu_streams_split():
    from suffix_hybrid.tools.helion_tune import gpu_streams
    assert gpu_streams("--tp 2 --families kgdn1") == [(None, ["--tp", "2", "--families", "kgdn1"])]
    two = gpu_streams("--families kgdn1,nvfp4_moe ;; --families fp8_moe")
    assert two == [(0, ["--families", "kgdn1,nvfp4_moe"]), (1, ["--families", "fp8_moe"])]


def test_raise_nofile_reaches_hard_limit():
    import resource
    from suffix_hybrid.tools.helion_tune import raise_nofile
    soft, hard = raise_nofile()
    assert soft == hard or hard == resource.RLIM_INFINITY
