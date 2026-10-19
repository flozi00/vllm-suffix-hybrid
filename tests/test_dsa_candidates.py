# SPDX-License-Identifier: Apache-2.0
"""Fused DeepSeek-V4.1 candidate selection (suffix_hybrid/kernels/dsa_candidates.py).

The anchor is checked against the vLLM nightly source it was cut from
(tests/fixtures), and the whole decode tail runs on CPU inside a stand-in
module whose sparse_attn_indexer IS that tail verbatim: vLLM's path (dense
logits -> candidate mask -> top-k) and the fused one select the same
positions, the mask is skipped only for the fused tensor, and nothing fuses
on the candidate SOURCE layer or with the gate off.
"""
import importlib.util
import os
import sys
import textwrap
from pathlib import Path

import pytest

os.environ.setdefault("HELION_INTERPRET", "1")
torch = pytest.importorskip("torch")
from suffix_hybrid import helion_kernels as hk  # noqa: E402
from suffix_hybrid.helion_kernels import dsa_indexer as hdi  # noqa: E402
from suffix_hybrid.kernels import dsa_candidates as dc  # noqa: E402
from suffix_hybrid.kernels import dsa_indexer as di  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import test_helion_dsa_indexer as th  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_39983a5e7" / "sparse_attn_indexer.py.txt"
NEG = float("-inf")

# The nightly decode tail (sparse_attn_indexer.py:880-946), verbatim, inside a
# function whose arguments are the locals it reads.
TAIL = '''
def sparse_attn_indexer(padded_q_quant_cast, kv_cache, decode_weights, seq_lens, decode_metadata,
                        max_model_len, candidate_blocks, candidate_block_size, candidate_write,
                        topk_backend, dcp_world_size, topk_indices_buffer, num_padded_tokens,
                        topk_tokens, next_n, attn_metadata_narrowed):
    padded_q_scale, dcp_rank, cp_kv_cache_interleave_size = None, 0, 1
    if True:
        if True:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                decode_weights,
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                indices=decode_metadata.indices,
            )
            # Keeping bf16 halves top-k/candidate reads; other top-k backends and
            # DCP need fp32. SM100 decode is varlen: one context length per row.
            if topk_backend != "deep_select" or dcp_world_size != 1:
                logits = as_fp32_logits(logits, seq_lens.view(-1))
        num_rows = logits.shape[0]
        if candidate_blocks is not None:
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            vis = vis[:num_rows]
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    None,
                    vis,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                    row_repeat,
                )
            else:
                _apply_candidate_mask(
                    logits,
                    None,
                    vis,
                    decode_candidates,
                    candidate_block_size,
                    row_repeat,
                )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )
    return topk_indices_buffer
'''


def test_anchor_is_the_nightly_decode_tail():
    src = FIXTURE.read_text()
    assert dc.tail_matches(src)
    assert dc.tail_matches(TAIL)
    # any extra consumer of the logits between the call and the top-k -> drift
    line = "row_repeat = next_n if vis.numel() != num_rows else 1\n"
    bad = src.replace(line, line + "            logits.mul_(1)\n", 1)
    assert bad != src and not dc.tail_matches(bad)
    i = src.rindex("_apply_candidate_mask(\n")   # the decode one (prefill's comes first)
    assert not dc.tail_matches(src[:i] + "_apply_mask2(\n" + src[i + 22:])


def _module(tmp_path, calls):
    """A stand-in for vllm's sparse_attn_indexer module: the tail above plus
    torch twins of the ops it calls (dense logits = the f64 spec)."""
    path = tmp_path / "fake_sai.py"
    path.write_text(textwrap.dedent(TAIL))
    spec = importlib.util.spec_from_file_location("fake_sai", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def dense(q, kv_cache, w, cl, bt, sched, max_model_len, indices=None):
        calls.append("dense")
        kv, ks, page, prow, sp = hdi.cache_views(kv_cache)
        rows = q.shape[0]
        val, _ = hdi.decode_reference(q.reshape(rows, -1, 128), w, kv, ks, bt, cl.reshape(-1),
                                      torch.empty((rows, max_model_len)), page, prow, sp, 2)
        return torch.where(torch.isnan(val), NEG, val).float()

    paged = di.wrap_paged(lambda *a, **k: pytest.fail("Triton fallback"))
    # vllm.utils.deep_gemm's wrapper sits between the indexer and the shim
    mod.fp8_fp4_paged_mqa_logits = lambda q, kv, w, cl, bt, sched, max_model_len, indices=None: \
        paged(q, kv, w, cl, bt, sched, max_model_len, indices=indices)
    mod._dense = dense
    mod.as_fp32_logits = lambda logits, ends: logits
    mod._select_candidate_blocks = lambda *a: calls.append("select")

    def mask(logits, starts, vis, cand, bsz, row_repeat):
        calls.append("mask")
        logits.copy_(th._vllm_mask(logits, vis, cand, bsz))
    mod._apply_candidate_mask = mask

    def get_topk(backend):
        def run(logits, seq_lens, next_n, topk_indices, k, max_seq_len):
            calls.append(("topk", logits.shape[1], max_seq_len))
            topk_indices.copy_(th._topk(logits, seq_lens.reshape(-1), k))
        return run
    mod.get_indexer_topk = get_topk
    mod._merge_dcp_topk_global = lambda *a: None
    return mod


class _Meta:
    def __init__(self, bt):
        self.block_table, self.schedule_metadata, self.indices = bt, None, None
        self.global_seq_lens = None
        self.max_seq_len = 400


def _run(mod, args, k, write=False):
    q, w, kv, ks, bt, cl, cand, out, pg, prow, sp, _ = args
    rows = q.shape[0]
    buf = torch.full((rows + 2, k), -7, dtype=torch.int32)
    cache = th.cache_of(kv, pg, prow).unsqueeze(2)
    mml = int(cl.max()) + 8   # vLLM: every context fits max_model_len
    return mod.sparse_attn_indexer(q.view(rows, 1, -1, 128), cache, w, cl.view(rows, 1), _Meta(bt),
                                   mml, cand, cand.shape[1] and out.shape[1] // cand.shape[1],
                                   write, "auto", 1, buf, rows, k, 1, _Meta(bt))[:rows]


@pytest.mark.parametrize("prob", th.CAND_PROBLEMS)
def test_fused_equals_vllm_path(monkeypatch, tmp_path, prob):
    heads, lens, nn, ncand, bsz, page, pad = prob
    args = th.cand_problem(heads, lens, nn, ncand, bsz, page, pad, seed=5)
    calls = []
    mod = _module(tmp_path, calls)
    monkeypatch.setattr(di, "paged_logits", lambda q, kv, w, cl, bt, mml, clean=False, **kw: mod._dense(
        q, kv, w, cl, bt, None, mml))
    monkeypatch.setattr(di, "paged_route", lambda q, kv, ctx=None: "oxide")  # dense = the spec above
    monkeypatch.setattr(dc, "_state", {"code": None, "pending": None, "sorted": None})
    monkeypatch.setattr(dc, "cands_sorted", lambda: True)
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"block_sizes": [32]} if kernel == hdi.CANDK else None)
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", th.make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)

    monkeypatch.delenv(dc.GATE, raising=False)
    want = _run(mod, args, 8)
    assert calls[:3] == ["dense", "mask", ("topk", max(lens) + 8, 400)]

    calls.clear()
    monkeypatch.setenv(dc.GATE, "1")
    dc.install(mod)
    got = _run(mod, args, 8)
    assert "dense" not in calls and "mask" not in calls          # fused, mask skipped
    assert calls == [("topk", ncand * bsz, min(ncand * bsz, 400))]
    assert dc._state["pending"] is None
    cl = args[5]
    for r in range(cl.shape[0]):
        assert {i for i in got[r].tolist() if i >= 0} == {i for i in want[r].tolist() if i >= 0}, r

    calls.clear()  # the candidate SOURCE layer stays dense (it selects the blocks)
    _run(mod, args, 8, write=True)
    assert calls[:2] == ["dense", "select"]


def test_gate_refuses_a_drifted_indexer(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "_state", {"code": None, "pending": None, "sorted": None})
    path = tmp_path / "drift.py"
    line = "row_repeat = next_n if vis.numel() != num_rows else 1\n"
    path.write_text(textwrap.dedent(TAIL).replace(line, line + "            logits.mul_(1)\n", 1))
    spec = importlib.util.spec_from_file_location("drift", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._apply_candidate_mask = mod.get_indexer_topk = None
    with pytest.raises(RuntimeError, match="drifted from the anchor"):
        dc.install(mod)


def test_unranked_fused_logits_raise(monkeypatch):
    monkeypatch.setattr(dc, "_state", {"code": object(), "pending": ("x",), "sorted": True})
    monkeypatch.setenv(dc.GATE, "1")
    with pytest.raises(RuntimeError, match="never ranked"):
        dc.maybe_fuse(None, None, None, None, None)
    assert dc._state["pending"] is None


def test_fusion_does_not_depend_on_the_steps_context(monkeypatch, tmp_path):
    """The call sits in a decode graph captured once (max_seq_len = max_model_len at
    capture): short and long contexts take the same path -- fused whenever the gate,
    the layer type and a shipped config say so."""
    heads, lens, nn, ncand, bsz, page, pad = th.CAND_PROBLEMS[0]
    args = th.cand_problem(heads, lens, nn, ncand, bsz, page, pad, seed=5)
    calls = []
    mod = _module(tmp_path, calls)
    monkeypatch.setattr(di, "paged_logits", lambda q, kv, w, cl, bt, mml, clean=False, **kw: mod._dense(
        q, kv, w, cl, bt, None, mml))
    monkeypatch.setattr(di, "paged_route", lambda q, kv, ctx=None: "oxide")
    monkeypatch.setattr(dc, "_state", {"code": None, "pending": None, "sorted": None})
    monkeypatch.setattr(dc, "cands_sorted", lambda: True)
    monkeypatch.setattr(hk, "config", lambda kernel, case, plat=None:
                        {"block_sizes": [32]} if kernel == hdi.CANDK else None)
    monkeypatch.setattr(hdi, "_WARM", set())
    monkeypatch.setattr(hdi, "hl", th.make_stub(32, {"min_tiles": 1, "split_div": 1}))
    monkeypatch.setattr(hdi, "PowerOfTwoFragment", lambda *a: None)
    monkeypatch.setattr(hk, "compiled", lambda body, kernel, case: body)
    monkeypatch.setenv(dc.GATE, "1")
    dc.install(mod)
    for meta_len in (1, 400, 10 ** 6):                       # whatever the metadata's max_seq_len says
        calls.clear()
        _Meta.max_seq_len = meta_len
        _run(mod, args, 8)
        assert "dense" not in calls and "mask" not in calls
    _Meta.max_seq_len = 400


def test_remap_launches_go_through_the_capture_guard(monkeypatch):
    """cand_remap's Triton launch is refused inside a capture until a warm-up ran it
    eagerly (the CPU path is the torch twin and needs no guard)."""
    import types
    from suffix_hybrid.kernels import dsa_capture as cap
    cap.reset()
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            return lambda *a, **kw: launches.append((grid, kw))
    fake_triton = types.SimpleNamespace(jit=lambda f: FakeKernel(), cdiv=lambda a, b: -(-a // b))
    fake_tl = types.SimpleNamespace(constexpr=int)
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_tl)
    fake_triton.language = fake_tl
    monkeypatch.setattr(dc, "_REMAP", {})

    class OnCuda(torch.Tensor):
        is_cuda = True
    idx = torch.zeros((4, 512), dtype=torch.int32).as_subclass(OnCuda)
    cand, cl = torch.zeros((4, 2048), dtype=torch.int32), torch.ones(4, dtype=torch.int32)
    monkeypatch.setattr(cap, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="inside a CUDA-graph capture"):
        dc.cand_remap(idx, cand, cl, 8)
    monkeypatch.setattr(cap, "capturing", lambda: False)
    dc.cand_remap(idx, cand, cl, 8)                           # the eager warm-up
    monkeypatch.setattr(cap, "capturing", lambda: True)
    dc.cand_remap(idx, cand, cl, 8)                           # same classes: allowed in the capture
    with pytest.raises(RuntimeError):                         # another block size = another binary
        dc.cand_remap(idx, cand, cl, 1)
    assert [g for g, kw in launches] == [(4, 1), (4, 1)] and launches[0][1] == {"BSZ": 8, "BLOCK": 512}
    cap.reset()


# ---------------------------------------------------------------------------
# candidate SOURCE layer: valid-block scoring + persistent top-k
# ---------------------------------------------------------------------------
class _ScoresKernel:
    """torch twin of candidate_blocks._block_scores_kernel (vLLM 39983a5e7)."""
    def __init__(self):
        self.full_width_seen = []

    def fn(self, logits, starts, ends, scores, valid_blocks, stride_row, stride_col, stride_start,
           stride_end, stride_scores, width, nblocks, BLOCK_SIZE, HAS_STARTS, ROW_REPEAT, TILE,
           FULL_WIDTH):
        raise AssertionError

    def __getitem__(self, grid):
        def launch(logits, starts, ends, scores, valid_blocks, *rest, TILE, FULL_WIDTH):
            width, nblocks, bsz, has_starts, repeat = rest[-5:]
            self.full_width_seen.append(FULL_WIDTH)
            if not FULL_WIDTH:
                scores.fill_(float("nan"))      # new_empty: the unscored tail is garbage
            for row in range(logits.shape[0]):
                start = int(starts[row // repeat]) if has_starts else 0
                end = int(ends[row // repeat])
                col_end = min(end, width)
                nvalid = -(-max(col_end - start, 0) // bsz)
                valid_blocks[row] = nvalid
                for b in range(nblocks if FULL_WIDTH else nvalid):
                    cols = torch.arange(start + b * bsz, start + (b + 1) * bsz)
                    v = torch.where(cols < col_end, logits[row, cols.clamp_max(width - 1)], torch.tensor(NEG))
                    sc = v.max()
                    if end > start and b == (end - start - 1) // bsz:
                        sc = torch.tensor(float("inf"))
                    scores[row, b] = sc
        return launch


def _stock_select(logits, ks, ke, k, bsz, out, repeat=1):
    """vLLM's non-DeepSelect branch: FULL_WIDTH scores + torch.topk (the tie rule)."""
    rows, width = logits.shape
    nb = -(-width // bsz)
    scores = torch.full((rows, nb), NEG)
    for r in range(rows):
        end = int(ke[r // repeat])
        start = int(ks[r // repeat]) if ks is not None else 0
        for b in range(-(-max(min(end, width) - start, 0) // bsz)):
            cols = torch.arange(start + b * bsz, min(start + (b + 1) * bsz, min(end, width)))
            sc = logits[r, cols].max()
            scores[r, b] = float("inf") if b == (end - start - 1) // bsz else sc
    top = scores.topk(min(k, nb), dim=-1)
    idx = torch.where(top.values != NEG, top.indices, -1).to(torch.int32)
    out.fill_(-1)
    out[:, :idx.shape[1]] = idx


def _persistent(scores, lens, next_n, out, k, bound):
    """persistent_topk contract: rows <= k long return 0..len-1, else the top-k of
    [0, len); -1 fill. Reads nothing past len (the tail here is poison)."""
    for r, n in enumerate(lens.reshape(-1).tolist()):
        out[r].fill_(-1)
        out[r, :min(n, k)] = (torch.arange(n) if n <= k else scores[r, :n].topk(k).indices).to(torch.int32)[:k]


@pytest.mark.parametrize("starts", [False, True])
@pytest.mark.parametrize("repeat", [1, 3])
def test_candidate_block_selection_matches_stock_and_skips_the_tail(monkeypatch, starts, repeat):
    import types
    monkeypatch.setattr(dc, "PERSISTENT_K", (64, 100))   # tiny stand-in for 512 / 1024 / 2048
    g = torch.Generator().manual_seed(7)
    rows, width, bsz, k = 6, 1200, 8, 64
    logits = torch.randn((rows, width), generator=g)
    ke = torch.tensor([width, 300, 9, 640, 1200, 0] if repeat == 1 else [width, 640], dtype=torch.int32)
    ks = torch.tensor([0, 5, 0, 17, 0, 0][:ke.shape[0]], dtype=torch.int32) if starts else None
    scores_kernel = _ScoresKernel()
    cb = types.SimpleNamespace(_block_scores_kernel=scores_kernel)
    poison = {}

    def get_topk(backend):
        assert backend == "persistent"
        return lambda scores, lens, nn, out, kk, bound: (
            poison.update(valid=lens.clone(), nblocks=scores.shape[1]), _persistent(scores, lens, nn, out, kk, bound))[1]
    stock_calls = []
    fast = dc.select_blocks_fn(cb, get_topk, lambda *a: stock_calls.append(a))
    want = torch.empty((rows, k), dtype=torch.int32)
    got = torch.full((rows, k), -9, dtype=torch.int32)
    _stock_select(logits, ks, ke, k, bsz, want, repeat)
    fast(logits, ks, ke, k, bsz, got, repeat)
    assert not stock_calls
    assert scores_kernel.full_width_seen == [False]               # valid blocks only
    for r in range(rows):
        w_, g_ = want[r].tolist(), got[r].tolist()
        assert sorted(x for x in w_ if x >= 0) == sorted(x for x in g_ if x >= 0), r
        assert g_ == sorted(g_, key=lambda x: x < 0) or all(x >= 0 for x in g_[:sum(x >= 0 for x in g_)])  # valid first
        assert -9 not in g_
    # outside the fast path (k not a persistent size, too few blocks, wrong dtype): the original runs
    fast(logits, ks, ke, 128, bsz, got[:, :1].new_empty((rows, 128)), repeat)
    fast(logits[:, :200], ks, ke, 64, bsz, got.new_empty((rows, 64)), repeat)
    fast(logits.double(), ks, ke, 64, bsz, got, repeat)
    assert len(stock_calls) == 3


def test_candsel_arming_checks_the_scores_kernel_signature(monkeypatch, tmp_path):
    import types
    monkeypatch.setattr(dc, "_state", {"code": None, "pending": None, "sorted": None})
    mod = _module(tmp_path, [])
    sk = _ScoresKernel()
    cbm = types.SimpleNamespace(_block_scores_kernel=sk)
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    for name in ("vllm.model_executor", "vllm.model_executor.kernels", "vllm.model_executor.kernels.attention",
                 "vllm.model_executor.kernels.attention.dsa"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.kernels.attention.dsa.candidate_blocks", cbm)
    sys.modules["vllm.model_executor.kernels.attention.dsa"].candidate_blocks = cbm
    monkeypatch.setenv(dc.SEL_GATE, "1")
    monkeypatch.delenv(dc.GATE, raising=False)
    monkeypatch.setattr(dc, "select_blocks_fn", lambda cb, get_topk, stock: "patched")
    stock = mod._select_candidate_blocks
    dc.install(mod)
    assert mod._select_candidate_blocks == "patched" and mod._apply_candidate_mask is not None
    assert not hasattr(mod, "_ARMED")
    # a drifted kernel signature refuses
    monkeypatch.setattr(dc, "_state", {"code": None, "pending": None, "sorted": None})
    sk.fn = lambda logits, starts, ends: None
    with pytest.raises(RuntimeError, match="_block_scores_kernel"):
        dc.install(mod)
    assert stock is not None


def test_cand_remap_table_form_covers_the_threshold_select_pipeline():
    """bsz 1 turns the candidate remap into a plain slot -> position table
    lookup (compacted survivors): dead slots / positions past the context -> -1."""
    tab = torch.tensor([[40, 7, 99, 3]], dtype=torch.int32)
    idx = torch.tensor([[2, 0, -1, 3]], dtype=torch.int32)
    out = dc.cand_remap(idx, tab, torch.tensor([50], dtype=torch.int32), 1)
    assert out.tolist() == [[-1, 40, -1, 3]]


def test_arming_never_happens_inside_a_capture(monkeypatch):
    """First-use state binding (patching vLLM's module) is refused while a capture is open."""
    from suffix_hybrid.kernels import dsa_capture as cap
    monkeypatch.setattr(dc, "_state", {"code": None, "pending": None, "sorted": None})
    monkeypatch.setenv(dc.GATE, "1")
    monkeypatch.setattr(cap, "capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not armed"):
        dc.maybe_fuse(None, None, None, None, None)
