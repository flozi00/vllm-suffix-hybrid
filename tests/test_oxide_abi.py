# SPDX-License-Identifier: Apache-2.0
"""CPU guard for the cuda-oxide kernel ABI: the host launch argument lists
(src/nvfp4_attn_oxide.rs, src/oxide.rs) must match the #[kernel] parameter
lists (kernels-oxide/*/src/main.rs) in count and 32/64-bit kind — a drift
would feed the GPU garbage parameters without any compile error."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def kernel_params(src: str, name: str):
    src = re.sub(r"//[^\n]*", "", src)
    m = re.search(rf"pub fn {name}\((.*?)\)\s*\{{", src, re.S)
    assert m, name
    kinds = []
    for p in filter(None, (x.strip() for x in m.group(1).split(","))):
        ty = p.split(":", 1)[1].strip()
        if ty.startswith("*"):
            kinds.append("Ptr")
        elif ty.startswith("&[") or ty.startswith("DisjointSlice"):
            kinds += ["Ptr", "Ptr"]  # (ptr, len) pair
        else:
            kinds.append({"u32": "U32", "i32": "I32", "f32": "F32"}[ty])
    return kinds


def host_args(src: str, var: str):
    src = re.sub(r"//[^\n]*", "", src)
    body = re.search(rf"let {var} = vec!\[(.*?)\];", src, re.S).group(1)
    out = []
    for tok in re.findall(r"Arg::(\w+)\(|\bu\(", body):
        out.append(tok or "U32")
    return out


def test_k2_launch_args_match_kernel_params():
    dev = (ROOT / "kernels-oxide/k2_nvfp4_attn/src/main.rs").read_text()
    host = (ROOT / "src/nvfp4_attn_oxide.rs").read_text()
    assert host_args(host, "partial_args") == kernel_params(dev, "nvfp4_attn_partial")
    assert host_args(host, "merge_args") == kernel_params(dev, "nvfp4_attn_merge")


def test_nvfp4_ds_mla_launch_args_match_kernel_params():
    dev = (ROOT / "kernels-oxide/nvfp4_ds_mla/src/main.rs").read_text()
    host = (ROOT / "src/nvfp4_ds_mla_oxide.rs").read_text()
    assert host_args(host, "partial_args") == kernel_params(dev, "nvfp4_ds_mla_attn_partial")
    assert host_args(host, "merge_args") == kernel_params(dev, "nvfp4_ds_mla_attn_merge")
    assert host_args(host, "args") == kernel_params(dev, "nvfp4_ds_mla_quant_store")


def test_probe_launch_args_match_kernel_params():
    dev = (ROOT / "kernels-oxide/probe/src/main.rs").read_text()
    host = (ROOT / "src/oxide.rs").read_text()
    call = re.search(r"&\[\s*(Arg::Ptr\(a_ptr\).*?)\]", host, re.S).group(1)
    assert re.findall(r"Arg::(\w+)\(", call) == kernel_params(dev, "oxide_probe")


def test_every_kernel_crate_pins_the_same_cuda_oxide_rev():
    wf = (ROOT / ".github/workflows/native.yml").read_text()
    rev = re.search(r"CUDA_OXIDE_REV: (\w+)", wf).group(1)
    crates = list((ROOT / "kernels-oxide").glob("*/Cargo.toml"))
    assert crates
    for c in crates:
        text = c.read_text()
        assert "[workspace]" in text, c
        assert set(re.findall(r'rev = "(\w+)"', text)) == {rev}, c


def test_ptx_facts_records_entry_param_counts():
    import importlib.util
    spec = importlib.util.spec_from_file_location("oxide_build", ROOT / "scripts/oxide_build.py")
    ob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ob)
    ptx = (".version 8.7\n.target sm_120\n"
           ".visible .entry a(\n\t.param .u64 a_param_0,\n\t.param .u32 a_param_1\n)\n{\n}\n"
           ".visible .entry b()\n{\n}\n")
    isa, target, entries, params = ob.ptx_facts(ptx)
    assert entries == ["a", "b"] and params == {"a": 2, "b": 0}


def test_k2_host_param_counts_match_kernel():
    """own_attn.K2_PARAMS (what prepare() demands of the loaded cubin's
    manifest) == the #[kernel] parameter counts == the host arg lists."""
    import sys
    sys.path.insert(0, str(ROOT / "sm120"))
    from nvfp4_kv_patch import own_attn
    dev = (ROOT / "kernels-oxide/k2_nvfp4_attn/src/main.rs").read_text()
    assert own_attn.K2_PARAMS == {
        e: len(kernel_params(dev, e)) for e in ("nvfp4_attn_partial", "nvfp4_attn_merge")}


def test_stale_cubin_abi_is_refused(tmp_path, monkeypatch):
    """A cubin whose entry takes a different parameter count than the host
    passes (or a pre-ABI-record manifest) must fail closed at load: an old
    cubin would otherwise silently ignore / misread launch arguments."""
    import json
    import pytest
    from suffix_hybrid import oxide_kernels as ok
    monkeypatch.setattr(ok, "_MANIFEST", None)
    for params in (None, {"k": 24}):
        ent = {"name": "fam", "file": "fam.cubin", "sha256": "0", "entries": ["k"]}
        if params:
            ent["params"] = params
        (tmp_path / "manifest.json").write_text(json.dumps({"arch": "sm_120", "kernels": [ent]}))
        with pytest.raises(RuntimeError, match="params"):
            ok.ensure_loaded("fam", 0, str(tmp_path), params={"k": 25})
        # also when probe() already loaded the family (no params there)
        monkeypatch.setattr(ok, "_LOADED", {("fam", 0)})
        with pytest.raises(RuntimeError, match="params"):
            ok.ensure_loaded("fam", 0, str(tmp_path), params={"k": 25})
        monkeypatch.setattr(ok, "_LOADED", set())
