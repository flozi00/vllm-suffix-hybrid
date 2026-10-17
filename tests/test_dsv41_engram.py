import types

import pytest

from suffix_hybrid import dsv41_engram


def _module(tmp_path, text):
    f = tmp_path / "engram.py"
    f.write_text(text)
    return types.SimpleNamespace(__file__=str(f), __dict__={})


def test_shipped_source_compiles_and_reads_mxfp4():
    src = dsv41_engram.PATCHED.read_text()
    compile(src, "engram.py", "exec")
    assert "engram_embed_quant" in src and "PACKED_FP4" in src


def test_upstream_support_is_inert(tmp_path):
    assert dsv41_engram.apply(_module(tmp_path, "x = 'engram_embed_quant'\n")) is False


def test_drifted_base_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="re-port"):
        dsv41_engram.apply(_module(tmp_path, "x = 1\n"))


def test_block64_wraps_stock_off_sm120(monkeypatch):
    import sys
    from suffix_hybrid import dsv41_block64

    class Backend:
        @staticmethod
        def get_supported_kernel_block_sizes(kv_cache_spec=None):
            return [128]

    mod = types.ModuleType("vllm.v1.attention.backends.mla.indexer")
    mod.DeepseekV41IndexerBackend = Backend
    plat = types.SimpleNamespace(current_platform=types.SimpleNamespace(
        is_device_capability_family=lambda f: f == FAMILY[0]))
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.platforms", plat)
    FAMILY = [120]
    dsv41_block64.apply(mod)
    assert Backend.get_supported_kernel_block_sizes() == [64]
    FAMILY[0] = 100
    assert Backend.get_supported_kernel_block_sizes() == [128]
