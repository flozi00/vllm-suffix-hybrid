# SPDX-License-Identifier: Apache-2.0
"""CPU test for SUFFIX_ROCM_AFP4_CONFIGS wiring + afp4_tune JSON / candidates (no AITER/GPU).

Silicon check: boot gate afp4_tune prints one JSON per shape; once they ship in
suffix_hybrid/configs/afp4/, boot gate afp4_tune_shipped marks every tuned M "json" (AITER
resolved the plugin file) and a pod with the gate logs "[suffix rocm-patch] ACTIVE: AITER
GEMM-AFP4WFP4 configs from ... (<n> shapes)".
"""
import importlib
import json
import sys

import pytest

from suffix_hybrid import rocm_patches as rp
from suffix_hybrid.tools import afp4_tune as at

AFP4 = rp.PATCHES["SUFFIX_ROCM_AFP4_CONFIGS"]
STANDARD = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# aiter/ops/triton/utils/gemm_config_utils.py @ v0.1.24.post1, trimmed: get_gemm_config =
# _get_gemm_config_cached minus the input asserts / lru_cache, around the byte-exact anchor;
# load_config_json / resolve_config_dir as in config_utils (AITER_DIR = the arch directory).
FAKE = '''import json

STANDARD_M_BOUNDS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
AITER_DIR = None


def load_config_json(fpath, required=True):
    try:
        with open(fpath) as fh:
            return json.load(fh)
    except FileNotFoundError:
        if required:
            raise
        return None


def resolve_config_dir(op, config_name, backend="triton"):
    return f"{AITER_DIR}/{backend}/{op}/{config_name.lower().replace('-', '_')}"


def get_gemm_config(config_name, M, N=None, K=None, bounds=None, specialized_filename=None,
                    backend="triton", B=None):
    cfg_dir = resolve_config_dir("gemm", config_name, backend=backend)
    config_dict = load_config_json(f"{cfg_dir}/DEFAULT.json")
    specialized_suffixes = []
    if specialized_filename is not None:
        specialized_suffixes = [specialized_filename]
    elif N is not None and K is not None:
        if B is not None:
            specialized_suffixes.append(f"B={B}-N={N}-K={K}")
        specialized_suffixes.append(f"N={N}-K={K}")

    is_tuned = False
''' + AFP4.old + '''        if specialized_config is not None:
            config_dict, is_tuned = specialized_config, True
            break

    search_bounds = (
        bounds if bounds is not None else config_dict.get("M_BOUNDS", STANDARD_M_BOUNDS)
    )
    for bound in search_bounds:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned
    for bound in reversed(search_bounds):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned
    if "any" in config_dict:
        return dict(config_dict["any"]), False
    raise KeyError(M)
'''


def entry(tag: int) -> dict:  # an AITER bucket entry told apart by BLOCK_SIZE_M
    return dict.fromkeys(at.KEYS, 1) | {"BLOCK_SIZE_M": tag, "cache_modifier": None}


def put(path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))


def test_anchor_once_and_drift():
    out = rp.patch_source(AFP4, FAKE)
    assert AFP4.old not in out and AFP4.new in out
    compile(out, "<t>", "exec")
    with pytest.raises(RuntimeError, match="drifted"):
        rp.patch_source(AFP4, out)


def test_plugin_json_first_else_aiter(tmp_path, monkeypatch):
    aiter, plugin = tmp_path / "aiter", tmp_path / "plugin"
    gemm = aiter / "triton" / "gemm"
    default = {"M_LEQ_8": entry(8), "M_LEQ_64": entry(64), "M_LEQ_512": entry(512),
               "any": entry(256)}
    put(gemm / "gemm_afp4wfp4" / "DEFAULT.json", default)
    put(gemm / "gemm_afp4wfp4" / "GEMM-AFP4WFP4-N=13312-K=2560.json", {"any": entry(99)})
    put(gemm / "gemm_afp4wfp4" / "GEMM-AFP4WFP4-N=16384-K=2560.json", {"any": entry(98)})
    put(gemm / "gemm_a16wfp4" / "DEFAULT.json", {"any": entry(7)})
    put(aiter / "gluon" / "gemm" / "gemm_afp4wfp4" / "DEFAULT.json", {"any": entry(5)})
    # what afp4_tune prints between its JSON markers, pasted as a file
    doc = at.to_json({m: entry(1000 + m) for m in at.MS}, default, STANDARD)
    put(plugin / "GEMM-AFP4WFP4-N=2560-K=6144.json", at._dumps(doc))
    put(plugin / "GEMM-AFP4WFP4-N=16384-K=2560.json", {"any": entry(77)})
    put(plugin / "GEMM-A16WFP4-N=2560-K=6144.json", {"any": entry(66)})  # other family: not ours

    patch = AFP4._replace(target="fake_gemm_config_utils",
                          new=AFP4.new.replace(repr(rp.AFP4_DIR), repr(str(plugin))))
    assert patch.new != AFP4.new
    (tmp_path / "fake_gemm_config_utils.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(rp.PATCHES, "SUFFIX_ROCM_AFP4_CONFIGS", patch)
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    monkeypatch.setenv("SUFFIX_ROCM_AFP4_CONFIGS", "1")
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    assert rp.install_post_import_hook()
    try:
        mod = importlib.import_module("fake_gemm_config_utils")
        mod.AITER_DIR = str(aiter)
        assert getattr(mod, rp._MARK)

        def tag(*args, **kw):
            cfg, tuned = mod.get_gemm_config(*args, **kw)
            return cfg["BLOCK_SIZE_M"], tuned

        # Plugin file: one bucket per tuned M via its M_BOUNDS, AITER's entries above 256.
        for m, want in ((1, 1001), (2, 1005), (5, 1005), (6, 1008), (33, 1040), (40, 1040),
                        (41, 1064), (129, 1160), (161, 1256), (256, 1256)):
            assert tag("GEMM-AFP4WFP4", m, 2560, 6144) == (want, True), m
        assert tag("GEMM-AFP4WFP4", 300, 2560, 6144) == (512, True)  # AITER's M_LEQ_512
        assert tag("GEMM-AFP4WFP4", 8192, 2560, 6144) == (256, False)  # AITER's "any"
        assert tag("GEMM-AFP4WFP4", 40, 16384, 2560) == (77, False)  # plugin beats AITER's file
        # No plugin file: AITER exactly as before.
        assert tag("GEMM-AFP4WFP4", 40, 13312, 2560) == (99, False)  # AITER's own N/K file
        assert tag("GEMM-AFP4WFP4", 5, 7, 64) == (8, False)  # AITER's DEFAULT.json
        assert tag("GEMM-AFP4WFP4", 40, 7, 64) == (64, False)
        # Other families and the gluon backend never read the plugin directory.
        assert tag("GEMM-A16WFP4", 40, 2560, 6144) == (7, False)
        assert tag("GEMM-AFP4WFP4", 40, 2560, 6144, backend="gluon") == (5, False)
    finally:
        sys.modules.pop("fake_gemm_config_utils", None)


def test_json_format():
    default = {"M_LEQ_8": entry(8), "M_LEQ_512": entry(512), "M_GEQ_4096": entry(4096),
               "any": entry(256)}
    doc = json.loads(at._dumps(at.to_json({m: entry(m) | {"extra": 0} for m in at.MS},
                                          default, STANDARD)))
    bounds = doc.pop("M_BOUNDS")
    assert bounds == sorted(set(at.MS) | set(STANDARD)) and all(type(b) is int for b in bounds)
    assert list(doc) == [f"M_LEQ_{m}" for m in at.MS] + ["M_LEQ_512", "M_GEQ_4096", "any"]
    assert all(list(v) == list(at.KEYS) for v in doc.values())  # AITER's keys only, in order


def test_candidates_follow_the_pruning_rules():
    base = dict.fromkeys(at.KEYS, 1) | {"num_warps": 4, "num_stages": 3, "waves_per_eu": 2,
                                        "matrix_instr_nonkdim": 32, "cache_modifier": ".cg"}

    def get_splitk(kb, bk, ks):  # AITER keeps a split that cuts K into BLOCK_K multiples
        return (2 * kb // ks, bk, ks) if (2 * kb) % (ks * bk) == 0 else (2 * kb, bk, 1)

    total = 0
    for n, k, _ in at.SHAPES:
        for m in at.MS:
            cands = at.stage_a(m, n, k, base, 128, get_splitk)
            total += len(cands)
            assert cands and len({at._key(c) for c in cands}) == len(cands)
            for c in cands:
                bm, bk, ks = c["BLOCK_SIZE_M"], c["BLOCK_SIZE_K"], c["NUM_KSPLIT"]
                tiles = -(-m // bm) * -(-n // c["BLOCK_SIZE_N"])
                assert list(c) == list(base) and c["matrix_instr_nonkdim"] == 16
                assert ks * tiles >= 64
                assert k % bk == 0 if ks == 1 else (
                    tiles < 256 and ks * m * 16 <= k and k % (ks * bk) == 0)
                assert c["GROUP_SIZE_M"] == 1 or (ks == 1 and m > bm)
                assert (c["num_stages"] == 1) == (k <= ks * bk)
                for v in at.stage_b(c, k):
                    assert sum(v[key] != c[key] for key in at.KEYS) == 1
                    assert v["matrix_instr_nonkdim"] == 16 or min(bm, c["BLOCK_SIZE_N"]) >= 32
    assert total < 2500  # stage A for 3 shapes x 10 Ms: minutes of compile in the pool
