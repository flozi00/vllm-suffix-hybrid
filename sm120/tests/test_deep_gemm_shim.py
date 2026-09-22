# SPDX-License-Identifier: Apache-2.0
"""Contract + numerics tests for the sm120 deep_gemm shim.

CPU-safe by design: the pure-torch reference implementation, the
shape/edge-case validation, and the import-shadowing test run everywhere
(the CI ubuntu runner installs CPU torch). Anything needing Triton/CUDA
skips cleanly when unavailable.

The reference mirrors the wrapper contract of vLLM 0.30
``vllm/utils/deep_gemm.py`` and the semantics of
``sparse_attn_indexer.py`` / ``indexer_k_quant_and_cache_kernel``:

    logits[m, n] = sum_h relu(dot(q[m,h], kv[n])) * weights[m,h] * k_scale[n]

with columns outside [cu_seqlen_ks[m], cu_seqlen_ke[m]) masked to -inf iff
clean_logits. The paged variant reads the SPLIT cache layout: per physical
page, block_size*D fp8 bytes then block_size fp32 scales.
"""
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
# NOTE: triton is NOT import-skipped at module level: the shadowing tests
# run the shim in a SUBPROCESS and the shim itself must import without
# triton (API-server processes). Triton-dependent tests skip via
# _fallback()/needs_cuda instead.

SHIM_DIR = Path(__file__).resolve().parents[1] / "deep_gemm_shim"


# --------------------------------------------------------------------------
# Pure-torch reference (also the executable spec of the contract)
# --------------------------------------------------------------------------

def ref_mqa_logits(q_fp8, kv_fp8, k_scale, weights, ks, ke, clean):
    """Reference for fp8_fp4_mqa_logits (FP8 path). fp32 throughout."""
    q = q_fp8.to(torch.float32)          # [M,H,D]
    kv = kv_fp8.to(torch.float32)        # [N,D]
    M, H, D = q.shape
    N = kv.shape[0]
    logits = torch.zeros((M, N), dtype=torch.float32)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for h in range(H):
                dot = torch.dot(q[m, h], kv[n]).item()
                acc += max(dot, 0.0) * weights[m, h].item()
            logits[m, n] = acc * k_scale[n].item()
    if clean:
        for m in range(M):
            lo, hi = int(ks[m]), int(ke[m])
            logits[m, :lo] = float("-inf")
            logits[m, hi:] = float("-inf")
    return logits


def build_paged_cache(k_fp8, scales, block_size):
    """Pack [T,D] fp8 + [T] fp32 scales into the vLLM indexer K-cache:
    uint8 [num_blocks, block_size, D+4] with values in the first
    block_size*D bytes and scales in the trailing block_size*4 bytes."""
    D = k_fp8.shape[1]
    T = k_fp8.shape[0]
    num_blocks = math.ceil(T / block_size)
    cache = torch.zeros((num_blocks, block_size, D + 4), dtype=torch.uint8)
    k_bytes = k_fp8.view(torch.uint8)
    s_bytes = scales.to(torch.float32).view(torch.uint8)
    for t in range(T):
        b, off = divmod(t, block_size)
        cache[b, off, :D] = k_bytes[t]
        cache[b, off, D:] = s_bytes[t]
    return cache


def ref_paged_mqa_logits(q_fp8, cache, block_tables, context_lens, weights,
                         max_len, clean, next_n):
    """Reference for fp8_fp4_paged_mqa_logits (FP8 path, one scale/token).
    q_fp8: [rows,H,D], weights: [rows,H], context_lens: [rows] (flat).
    block_tables entries are PHYSICAL page indices, as delivered."""
    block_size = cache.shape[1]
    D = cache.shape[2] - 4
    q = q_fp8.to(torch.float32)
    rows = q.shape[0]
    out = torch.zeros((rows, max_len), dtype=torch.float32)
    for r in range(rows):
        b = r // next_n
        cl = int(context_lens[r])
        for n in range(min(cl, max_len)):
            blk, off = divmod(n, block_size)
            phys = int(block_tables[b, blk])
            k_bytes = cache[phys, off, :D]
            k = k_bytes.view(torch.float8_e4m3fn).to(torch.float32)
            k_scale = cache[phys, off, D:].view(torch.float32).item()
            acc = 0.0
            for h in range(q.shape[1]):
                dot = torch.dot(q[r, h], k).item()
                acc += max(dot, 0.0) * weights[r, h].item()
            out[r, n] = acc * k_scale
        if clean:
            out[r, cl:max_len] = float("-inf")
    return out


def _make_inputs(M=6, H=8, N=18, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    qf = (torch.randn((M, H, D), generator=g) * 0.3).clamp(-4, 4)
    kvf = (torch.randn((N, D), generator=g) * 0.3).clamp(-4, 4)
    q_fp8 = qf.to(torch.float8_e4m3fn)
    kv_fp8 = kvf.to(torch.float8_e4m3fn)
    k_scale = torch.rand(N, generator=g) * 0.05 + 0.01
    weights = torch.rand((M, H), generator=g)
    ks = torch.zeros(M, dtype=torch.int32)
    ke = torch.arange(1, M + 1).clamp(max=N).to(torch.int32)
    return q_fp8, kv_fp8, k_scale, weights, ks, ke


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not importlib.util.find_spec("triton"),
    reason="CUDA+Triton required for kernel numerics")


def _fallback():
    pytest.importorskip("triton")
    spec = importlib.util.spec_from_file_location(
        "sm120_fallback_under_test", SHIM_DIR / "sm120_fallback.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Import-shim resolution: a dir prepended to sys.path shadows site-packages
# (module-level importorskip above would skip these on triton-free runners,
# but the shim itself must import WITHOUT triton — API-server processes —
# so these two tests live in their own module loaded directly.)
# --------------------------------------------------------------------------

def test_shim_dir_shadows_site_packages(tmp_path):
    """Simulate the bundle layout: <tmp>/deep_gemm/ (our shim, package) must
    win over a decoy deep_gemm.py module earlier in the default path."""
    decoy = tmp_path / "site-packages"
    decoy.mkdir()
    (decoy / "deep_gemm.py").write_text("MARKER = 'site-packages'\n")
    shadow = tmp_path / "plugins"
    shadow.mkdir()
    (shadow / "deep_gemm").symlink_to(SHIM_DIR)
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys\n"
        "import deep_gemm\n"
        "assert getattr(deep_gemm, 'MARKER', None) != 'site-packages', "
        "'deep_gemm resolved to site-packages'\n"
        "assert getattr(deep_gemm, '__suffix_shim__', False), "
        "'deep_gemm did not resolve to the suffix shim'\n"
        "print('OK', deep_gemm.__file__)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shadow), str(decoy), env.get("PYTHONPATH", "")])
    # Sanity: WITHOUT the shadow dir the decoy wins (proves the decoy is real).
    env_plain = dict(env, PYTHONPATH=os.pathsep.join(
        [str(decoy), env.get("PYTHONPATH", "")]))
    r_plain = subprocess.run([sys.executable, str(probe)], env=env_plain,
                             capture_output=True, text=True)
    assert r_plain.returncode != 0, "decoy should have won without the shim"
    r = subprocess.run([sys.executable, str(probe)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_shim_import_is_side_effect_free_without_vendor(tmp_path):
    """On a machine with neither deep_gemm nor vllm installed, importing the
    shim must succeed cheaply and every probed symbol must raise loudly on
    call (never a silent None). Also proves NO torch import happens at
    module top level."""
    pkg_root = tmp_path / "plugins"
    pkg_root.mkdir()
    (pkg_root / "deep_gemm").symlink_to(SHIM_DIR)
    script = tmp_path / "probe2.py"
    script.write_text(
        "import sys\n"
        "import deep_gemm\n"
        "assert 'torch' not in sys.modules, "
        "'shim imported torch at module level'\n"
        "for name in ('fp8_gemm_nt', 'fp8_fp4_mqa_logits', 'fp8_einsum', "
        "'set_pdl', 'get_num_sms', 'mega_mhc', "
        "'get_paged_mqa_logits_metadata', 'fp8_fp4_paged_mqa_logits'):\n"
        "    fn = getattr(deep_gemm, name)\n"
        "    try:\n        fn()\n    except (RuntimeError, NotImplementedError, "
        "ValueError) as e:\n        assert 'unavailable' in str(e).lower(), "
        "name\n    else:\n        raise AssertionError(name + ' did not raise')\n"
        "print('OK')\n"
    )
    # torch IS importable here (pytest imported it) — block it via a meta
    # path hook so the probe sees a torch-free process.
    blocker = tmp_path / "blocktorch.py"
    blocker.write_text(
        "import sys\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'torch' or name.startswith('torch.'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "sys.modules.pop('torch', None)\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import blocktorch  # installs the torch blocker first\n"
        "import runpy\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(pkg_root), str(tmp_path), env.get("PYTHONPATH", "")])
    r = subprocess.run([sys.executable, str(driver)], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# --------------------------------------------------------------------------
# CPU-side contract validation of the fallback wrappers (no kernel launch)
# --------------------------------------------------------------------------

def test_fallback_import_skips_without_triton():
    pytest.importorskip("triton")


def test_metadata_shape_and_dtype():
    fb = _fallback()
    cl = torch.ones(3, dtype=torch.int32)
    meta = fb.get_paged_mqa_logits_metadata(cl, 64, 132)
    assert meta.shape == (133, 2) and meta.dtype == torch.int32


def test_mqa_rejects_fp4_inputs():
    fb = _fallback()
    packed = torch.zeros(2, 4, 64, dtype=torch.uint8)
    scale = torch.zeros(2, 4, 4, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="MXFP4"):
        fb.fp8_fp4_mqa_logits((packed, scale), (packed, scale),
                              torch.zeros(2, 4), torch.zeros(2, dtype=torch.int32),
                              torch.zeros(2, dtype=torch.int32), False)
    # Wrong dtype on the value tensors (e.g. bf16 sneak-in) is also loud.
    with pytest.raises(NotImplementedError, match="float8"):
        fb.fp8_fp4_mqa_logits(
            (torch.zeros(2, 4, 8), None),
            (torch.zeros(3, 8), torch.zeros(3)),
            torch.zeros(2, 4), torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32), False)


@needs_cuda
def test_mqa_logits_matches_reference():
    fb = _fallback()
    q, kv, k_scale, w, ks, ke = _make_inputs()
    q, kv, k_scale, w = (t.cuda() for t in (q, kv, k_scale, w))
    ks, ke = ks.cuda(), ke.cuda()
    got = fb.fp8_fp4_mqa_logits((q, None), (kv, k_scale), w, ks, ke,
                                clean_logits=True)
    want = ref_mqa_logits(q.cpu(), kv.cpu(), k_scale.cpu(), w.cpu(), ks.cpu(),
                          ke.cpu(), clean=True).cuda()
    # Products of fp8-exact values in fp32 accumulation: tolerance covers
    # accumulation order only.
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4)


@needs_cuda
def test_mqa_logits_no_clean_leaves_window_scores():
    fb = _fallback()
    q, kv, k_scale, w, ks, ke = _make_inputs(seed=1)
    q, kv, k_scale, w = (t.cuda() for t in (q, kv, k_scale, w))
    got = fb.fp8_fp4_mqa_logits((q, None), (kv, k_scale.cuda()), w,
                                ks.cuda(), ke.cuda(), clean_logits=False)
    assert torch.isfinite(got).all()
    want = ref_mqa_logits(q.cpu(), kv.cpu(), k_scale.cpu(), w.cpu(), ks, ke,
                          clean=False).cuda()
    M, N = got.shape
    for m in range(M):
        lo, hi = int(ks[m]), int(ke[m])
        torch.testing.assert_close(got[m, lo:hi], want[m, lo:hi],
                                   rtol=1e-3, atol=1e-4)


@needs_cuda
def test_paged_mqa_logits_matches_reference():
    fb = _fallback()
    g = torch.Generator().manual_seed(2)
    B, next_n, H, D, block_size = 3, 2, 8, 128, 4
    rows = B * next_n
    T = 11  # total K tokens across the (paged) context
    kf = (torch.randn((T, D), generator=g) * 0.3).clamp(-4, 4)
    k_fp8 = kf.to(torch.float8_e4m3fn)
    scales = torch.rand(T, generator=g) * 0.05 + 0.01
    cache_cpu = build_paged_cache(k_fp8, scales, block_size)
    # Out-of-order physical pages: every request reads its context from
    # physical pages [2, 1, 0] (logical block -> physical via block table).
    # Re-token the logical stream so physical scatter is meaningful: token t
    # lives at logical block t//bs -> physical 2 - t//bs.
    block_tables = torch.tensor([[2, 1, 0]], dtype=torch.int32).repeat(B, 1)
    # Build cache directly in physical order: page p holds tokens whose
    # logical block maps to p, i.e. logical block (2-p).
    phys_cache = torch.zeros_like(cache_cpu)
    for logical in range(3):
        physical = 2 - logical
        phys_cache[physical] = cache_cpu[logical]
    qf = (torch.randn((rows, H, D), generator=g) * 0.3).clamp(-4, 4)
    q_fp8 = qf.to(torch.float8_e4m3fn)
    weights = torch.rand((rows, H), generator=g)
    # Per-row context lengths (B, next_n) form, as native spec decode sends.
    cl2d = torch.tensor([[5, 6]] * B, dtype=torch.int32)
    max_len = 16
    cache = phys_cache.cuda()
    kv_view = cache.unsqueeze(2)  # sap.py's 4D [blocks, bs, 1, width] view
    got = fb.fp8_fp4_paged_mqa_logits(
        (q_fp8.cuda(), None), kv_view, weights.cuda(), cl2d.cuda(),
        block_tables.cuda(), None, max_len, clean_logits=True)
    want = ref_paged_mqa_logits(q_fp8, phys_cache, block_tables,
                                cl2d.reshape(-1), weights, max_len,
                                clean=True, next_n=next_n).cuda()
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------
# Shape/edge cases: zero-length windows, ks==ke rows, M < batch
# --------------------------------------------------------------------------

def test_edge_validation_zero_and_degenerate_windows():
    """CPU-side: shape checks fire before any launch; degenerate windows are
    representable and must not crash the validators."""
    pytest.importorskip("triton")
    fb = _fallback()
    D, H = 128, 8
    q = torch.zeros(4, H, D, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(7, D, dtype=torch.float8_e4m3fn)
    ks = torch.tensor([0, 3, 5, 5], dtype=torch.int32)
    ke = torch.tensor([0, 3, 5, 5], dtype=torch.int32)  # every row empty
    w = torch.ones(4, H)
    if not torch.cuda.is_available():
        pytest.skip("validators pass and launch needs CUDA")
    out = fb.fp8_fp4_mqa_logits((q.cuda(), None),
                                (kv.cuda(), torch.rand(7).cuda()),
                                w.cuda(), ks.cuda(), ke.cuda(), True)
    assert out.shape == (4, 7)
    assert torch.isneginf(out).all()  # every window empty -> all -inf


def test_zero_sized_batches():
    fb = _fallback()
    D, H = 128, 8
    q = torch.zeros(0, H, D, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(0, D, dtype=torch.float8_e4m3fn)
    if torch.cuda.is_available():
        q, kv = q.cuda(), kv.cuda()
    empty = torch.empty(0, dtype=torch.float32)
    out = fb.fp8_fp4_mqa_logits((q, None), (kv, empty), torch.zeros(0, H),
                                torch.zeros(0, dtype=torch.int32),
                                torch.zeros(0, dtype=torch.int32), False)
    assert out.shape == (0, 0)


def test_m_smaller_than_batch_semantics():
    """sap.py slices weights/ks/ke to the same token window as q: the shim
    must key everything off M (q rows), never an implied batch size."""
    fb = _fallback()
    D, H, N = 128, 8, 5
    q = torch.zeros(2, H, D, dtype=torch.float8_e4m3fn)  # M=2 < N=5
    kv = torch.zeros(N, D, dtype=torch.float8_e4m3fn)
    w = torch.ones(2, H)
    ks = torch.tensor([0, 1], dtype=torch.int32)
    ke = torch.tensor([5, 5], dtype=torch.int32)
    if not torch.cuda.is_available():
        pytest.skip("kernel launch needs CUDA")
    out = fb.fp8_fp4_mqa_logits((q.cuda(), None),
                                (kv.cuda(), torch.rand(N).cuda()),
                                w.cuda(), ks.cuda(), ke.cuda(), True)
    assert out.shape == (2, N)
    assert torch.isneginf(out[1]).all()          # ke==ks row fully masked
    assert torch.isfinite(out[0, :5]).all()


def test_weights_shape_mismatch_loud():
    fb = _fallback()
    D, H = 128, 8
    q = torch.zeros(2, H, D, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(3, D, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="weights"):
        fb.fp8_fp4_mqa_logits((q, None), (kv, torch.ones(3)),
                              torch.ones(H),  # wrong: [H] not [M,H]
                              torch.zeros(2, dtype=torch.int32),
                              torch.ones(2, dtype=torch.int32), False)
