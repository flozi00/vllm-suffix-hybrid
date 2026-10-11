# SPDX-License-Identifier: Apache-2.0
"""Own single-pass HiSparse resolver (hisparse_mtp_patch.own_resolver), CPU side:
the plain reference the GPU oracle checks our kernel against == a phase-by-phase
model of vLLM's hisparse_resolve_residency kernel (hisparse_kernels.cu:276-747),
with upstream's int32 union values AND our 16-bit ones; kernel parameter
buffers == the CUDA signatures; the nightly call sites route to our launchers
with upstream's arguments; the boot oracle fails closed."""
import inspect
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "sm120")]
import hisparse_mtp_patch as HS  # noqa: E402
from hisparse_mtp_patch import own_resolver as OR  # noqa: E402

RUNTIME_FIXTURE = Path(__file__).resolve().parent / "fixtures/vllm_0.31.1rc1.dev273/hisparse_runtime.py"
REAL_RESOLVE, REAL_INVALIDATE = OR.resolve_residency, OR.invalidate_written_slots


class Trap(Exception):
    pass


UPSTREAM = types.SimpleNamespace(  # int32 values: position, else 1<<30|slot, 1<<29|slot, 1<<28
    unset=2**31 - 1, hit=lambda s: 1 << 30 | s, miss=lambda s: 1 << 29 | s, invalid=1 << 28,
    resolved=lambda v: bool(v & (7 << 28)), is_invalid=lambda v: bool(v & 1 << 28),
    is_miss=lambda v: bool(v & 1 << 29), slot=lambda v: v & 0xFFFF, position_ok=lambda p: True)
OURS = types.SimpleNamespace(  # 16-bit values: position < 0x3FFF, else 2-bit tag | 14-bit slot
    unset=0x3FFF, hit=lambda s: 0x4000 | s, miss=lambda s: 0x8000 | s, invalid=0xC000,
    resolved=lambda v: (v & 0xC000) != 0, is_invalid=lambda v: (v & 0xC000) == 0xC000,
    is_miss=lambda v: (v & 0xC000) == 0x8000, slot=lambda v: v & 0x3FFF,
    position_ok=lambda p: p < 0x3FFF)


def kernel_model(enc, hot_bt, gidx, req_ids, state_idx, src_bt, src_bs, res_bt, res_bs, res_null,
                 host_rows, dgi, lru, hot_bs, attn_stride, union_rows):
    """hisparse_resolve_residency_kernel, one launch, phase by phase on Python
    lists (dgi / lru mutated). Thread interleavings do not change its results:
    a reference phase 3 leaves unresolved is written by phase 4 to the same
    slot, and every entry's first reference sees its position."""
    n, k, hot = len(gidx), len(gidx[0]), len(lru[0])
    hash_size, nbc = union_rows + 1, -(-hot // 32)
    out = {key: [[-9] * k for _ in range(n)] for key in ("hot", "attn", "swap_host", "swap_dev")}
    out.update(valid=[-9] * n, swap_cnt=[-9] * n, hits=0, misses=0)

    def translate(t, rid, st):
        sb = t // src_bs if t >= 0 else -1
        if 0 <= rid < len(src_bt) and 0 <= sb < len(src_bt[0]):
            pb = src_bt[rid][sb]
            g = pb * src_bs + t % src_bs if pb > 0 else -1
        else:
            g = -1
        resident = -1
        if res_bt is not None:
            rb = t // res_bs if t >= 0 else -1
            if 0 <= st < len(res_bt) and 0 <= rb < len(res_bt[0]):
                pb = res_bt[st][rb]
                if pb != res_null and pb >= 0:
                    resident = pb * res_bs + t % res_bs
        return (-1 if g >= host_rows else g), resident

    def phys(rid, s):
        return hot_bt[rid][s // hot_bs] * hot_bs + s % hot_bs

    def store(r, i, p):
        out["hot"][r][i] = p
        out["attn"][r][i] = -1 if p < 0 else (p // hot_bs) * attn_stride + p % hot_bs

    for first in range(n):  # one block per launch row; run leaders resolve
        rid = req_ids[first]
        if first > 0 and req_ids[first - 1] == rid:
            continue
        end = first + 1
        while end < n and req_ids[end] == rid:
            end += 1
        st = state_idx[rid] if 0 <= rid < len(state_idx) else -1
        if st < 0:
            for r in range(first, end):
                for i in range(k):
                    store(r, i, -1)
                out["valid"][r] = out["swap_cnt"][r] = 0
            continue
        vals, union = {}, 0  # the open-addressing table: host row -> value
        for r in range(first, end):  # phase 1
            count = 0
            for i in range(k):
                g, resident = translate(gidx[r][i], rid, st)
                count += resident >= 0 or g >= 0
                if resident >= 0:
                    store(r, i, resident)
                elif g < 0:
                    store(r, i, -1)
                else:
                    p = (r - first) * k + i
                    if g not in vals:
                        vals[g], union = enc.unset, union + 1
                    if enc.position_ok(p):
                        vals[g] = min(vals[g], p)
                    else:
                        union += hash_size
            out["valid"][r] = count
        if union >= hash_size:
            raise Trap("union above max_union_rows")
        if union == 0:
            for r in range(first, end):
                out["swap_cnt"][r] = 0
            continue
        if (end - first) * k > nbc * 32:
            raise Trap("positions above the buffer chunks")
        lru_out, H, E = [None] * hot, 0, 0  # phase 2
        for pos in range(hot):
            s = lru[st][pos]
            c = dgi[st][s] if 0 <= s < hot else -1
            if c >= 0 and c in vals:
                vals[c] = enc.hit(s)
                lru_out[H], H = s, H + 1
            else:
                lru_out[hot - 1 - E], E = s, E + 1
        M, loads, refs = 0, set(), []  # phase 3
        for p in range((end - first) * k):
            g, resident = translate(gidx[first + p // k][p % k], rid, st)
            if resident >= 0 or g < 0:
                continue
            refs.append((p, g))
            if vals[g] == p:
                ev, c = lru_out[hot - 1 - M], first * k + M
                out["swap_host"][c // k][c % k] = g
                if ev < 0 or ev >= hot:
                    vals[g] = enc.invalid
                    out["swap_dev"][c // k][c % k] = -1
                else:
                    vals[g] = enc.miss(ev)
                    out["swap_dev"][c // k][c % k] = phys(rid, ev)
                    dgi[st][ev] = g
                loads.add(p)
                M += 1
        for r in range(first, end):
            out["swap_cnt"][r] = max(0, min(k, M - (r - first) * k))
        for p, g in refs:  # phases 3 + 4: every host reference
            v = vals[g]
            assert enc.resolved(v)
            r, i = first + p // k, p % k
            if enc.is_invalid(v):
                store(r, i, -1)
                continue
            store(r, i, phys(rid, enc.slot(v)))
            if enc.is_miss(v) and p in loads:
                out["misses"] += 1
            else:
                out["hits"] += 1
        R = hot - H - M  # phase 5
        lru[st] = [lru_out[hot - 1 - M - i] if i < R else
                   lru_out[hot - 1 - (i - R)] if i < R + M else lru_out[i - R - M]
                   for i in range(hot)]
    return out


def _case(rnd, k, q_max, reqs, hot, ctx):
    """One launch: runs of 1..q_max rows per request (rows unique, ~70 % shared
    within a request, some -1), V2 padding (request -1), a masked request-0 run,
    unmapped requests, null / out-of-pool source blocks, an optional resident table."""
    bs = 4
    blocks = -(-ctx // bs)
    ids, rows = [], []
    order = list(range(reqs + 1))  # request `reqs` has no state
    rnd.shuffle(order)
    for rid in order[:rnd.randint(1, reqs + 1)]:
        base = rnd.sample(range(ctx), k)
        for _ in range(rnd.randint(1, q_max)):
            row = list(dict.fromkeys(t if rnd.random() < 0.7 else rnd.randrange(ctx) for t in base))
            have = set(row)
            row += [t for t in rnd.sample(range(ctx), min(ctx, 3 * k)) if t not in have][:k - len(row)]
            rows.append([t if rnd.random() > 0.05 else -1 for t in row])
            ids.append(rid)
    if rnd.random() < 0.3 and ids[-1] != 0:  # request 0 again after the real rows, masked
        ids += [0] * 2
        rows += [[-1] * k] * 2
    if rnd.random() < 0.3:  # V2 padding
        ids += [-1] * 3
        rows += [rnd.sample(range(ctx), k) for _ in range(3)]
    src_bt = [[0 if rnd.random() < 0.05 else rnd.randrange(1, 4 * reqs * blocks)  # 0: null block
               for _ in range(blocks)] for _ in range(reqs + 1)]
    res_bt = None
    if rnd.random() < 0.4:
        res_bt = [[rnd.choice([0, 0, 0, -2, rnd.randrange(1, 50)]) for _ in range(blocks)]
                  for _ in range(reqs + 1)]
    hot_bt = [rnd.sample(range(1, (reqs + 1) * hot // bs + 1), hot // bs) for _ in range(reqs + 1)]
    return dict(hot_bt=hot_bt, gidx=rows, req_ids=ids, src_bt=src_bt, src_bs=bs, res_bt=res_bt,
                res_bs=bs, res_null=0, host_rows=3 * reqs * blocks * bs, hot_bs=bs, attn_stride=7)


@pytest.mark.parametrize("k,q_max,extra,steps", [(8, 6, 0, 120), (4, 3, 8, 120), (16, 2, 32, 120),
                                                 (8, 1, 0, 120), (2048, 6, 0, 6)])  # last: GLM k=5
def test_reference_is_the_kernels_semantics(k, q_max, extra, steps):
    rnd = random.Random(k * 100 + q_max * 10 + extra)
    reqs = 4
    hot = (q_max + 1) * k + extra  # upstream default device_buffer_size = (q + 1) * top_k
    union_rows = q_max * k
    state_idx = [2, -1, 0, 3]  # request 1 is padding-mapped; state 1 unused
    dgi = torch.full((reqs, hot), -1, dtype=torch.int32)
    lru = torch.arange(hot, dtype=torch.int16).repeat(reqs, 1)
    models = {"upstream": ([[-1] * hot for _ in range(reqs)], [list(range(hot)) for _ in range(reqs)]),
              "ours": ([[-1] * hot for _ in range(reqs)], [list(range(hot)) for _ in range(reqs)])}
    for step in range(steps):
        c = _case(rnd, k, q_max, len(state_idx), hot, ctx=6 * k)
        if step % 17 == 16 or step == 4:  # a corrupted LRU entry: its eviction resolves invalid
            st, pos, bad = rnd.randrange(reqs), rnd.randrange(hot), rnd.choice([-1, hot + 3])
            lru[st, pos] = bad
            for _dgi, m_lru in models.values():
                m_lru[st][pos] = bad
        ref, dgi, lru = OR.reference_resolve(
            torch.tensor(c["hot_bt"]), torch.tensor(c["gidx"], dtype=torch.int32),
            torch.tensor(c["req_ids"]), torch.tensor(state_idx), torch.tensor(c["src_bt"]),
            c["src_bs"], None if c["res_bt"] is None else torch.tensor(c["res_bt"]), c["res_bs"],
            c["res_null"], c["host_rows"], dgi, lru, c["hot_bs"], c["attn_stride"])
        for name, enc in (("upstream", UPSTREAM), ("ours", OURS)):
            m_dgi, m_lru = models[name]
            got = kernel_model(enc, c["hot_bt"], c["gidx"], c["req_ids"], state_idx, c["src_bt"],
                               c["src_bs"], c["res_bt"], c["res_bs"], c["res_null"],
                               c["host_rows"], m_dgi, m_lru, c["hot_bs"], c["attn_stride"],
                               union_rows)
            for key in ("hot", "attn", "valid", "swap_cnt", "swap_host", "swap_dev"):
                assert ref[key].tolist() == got[key], (name, step, key)
            assert (ref["hits"], ref["misses"]) == (got["hits"], got["misses"]), (name, step)
            assert dgi.tolist() == m_dgi and lru.tolist() == m_lru, (name, step)


def test_union_above_the_bound_traps_like_upstream():
    hot_bt, src_bt = [[1, 2, 3, 4]], [[1, 2, 3, 4]]
    gidx = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 8 distinct host rows, bound 7
    for enc in (UPSTREAM, OURS):
        with pytest.raises(Trap, match="union"):
            kernel_model(enc, hot_bt, gidx, [0, 0], [0], src_bt, 4, None, 4, 0, 100,
                         [[-1] * 16], [list(range(16))], 4, 4, 7)


def test_glm_k5_fits_one_pass_on_sm120():
    hot, union, sm120 = 7 * 2048, 6 * 2048, 101376
    assert HS.residency_smem_bytes(union, hot) == 132388  # upstream: refused / split
    assert OR.smem_bytes(union, hot) == 79140 and OR.fits(union, hot, sm120)
    assert OR.fits(hot, hot, sm120)  # any union up to the whole hot buffer
    assert not OR.fits(8 * 2048, 8 * 2048, 1 << 30)  # 16,384 slots: past the 14-bit encoding


def _signature(name):
    params = OR.CUDA_SRC.split(f"void {name}(")[1].split(")")[0].split(",")
    return "@" + "".join("P" if "*" in p else "q" if "int64_t" in p else "i" for p in params)


def test_param_buffers_match_the_kernel_signatures():
    assert OR._RESOLVE_PARAMS.format == _signature("suffix_hisparse_resolve")
    assert OR._INVALIDATE_PARAMS.format == _signature("suffix_hisparse_invalidate")
    assert OR._RESOLVE_PARAMS.size == 272  # C layout: int64s 8-aligned after int32s


def test_patch_source_routes_both_op_calls_and_fails_closed_on_drift():
    src = RUNTIME_FIXTURE.read_text()
    methods = OR.patch_source(src)
    assert set(methods) == {"_resolve_residency", "invalidate_written_slots"}
    assert '_suffix_hisparse_op(self, "resolve")(' in methods["_resolve_residency"]
    assert '_suffix_hisparse_op(self, "invalidate")(' in methods["invalidate_written_slots"]
    assert all("torch.ops._C_cache_ops" not in m for m in methods.values())
    with pytest.raises(HS.PatchDriftError, match="invalidate_written_slots"):
        OR.patch_source(src.replace("hisparse_invalidate_written_slots(\n", "hisparse_invalidate(\n"))


def test_installed_methods_pass_upstreams_arguments(monkeypatch, tmp_path):
    rt = types.ModuleType("fake_hisparse_runtime")
    rt.__dict__.update(torch=torch, get_max_shared_memory_bytes=lambda index: 101376)

    class HiSparseRuntime:
        def __init__(self, config):
            self.device, self.is_group_leader = torch.device("cpu"), True

    rt.HiSparseRuntime = HiSparseRuntime
    OR.install(rt, OR.patch_source(RUNTIME_FIXTURE.read_text()), tmp_path / "runtime.py")
    calls = []
    monkeypatch.setattr(OR, "resolve_residency", lambda *a, **kw: calls.append(("r", a, kw)))
    monkeypatch.setattr(OR, "invalidate_written_slots", lambda *a, **kw: calls.append(("i", a, kw)))

    n, k, hot = 6, 8, 64
    i32 = dict(dtype=torch.int32)
    shared = types.SimpleNamespace(**{f: torch.zeros((n, k), **i32) for f in (
        "device_topk_rows", "physical_topk_indices", "swap_host_physical_rows",
        "swap_device_physical_rows")}, swap_counts=torch.zeros(n, **i32),
        valid_topk_counts=torch.zeros(n, **i32))
    group = types.SimpleNamespace(shared_topk=shared, device_global_indices=torch.zeros((2, hot), **i32),
                                  lru_slots=torch.zeros((2, hot), dtype=torch.int16),
                                  swap_stats=torch.zeros(2, dtype=torch.int64),
                                  _suffix_lru_scratch=torch.zeros((2, hot), dtype=torch.int16))
    inst = HiSparseRuntime(None)
    inst.__dict__.update(
        hot=types.SimpleNamespace(block_size=4, cache=torch.zeros(1), attention_block_stride=9),
        host_cache=torch.zeros(1), request_state_indices=torch.arange(2, **i32), index_group=group,
        hot_block_table=torch.zeros((2, 16), **i32), region_stride=hot, max_union_rows=48)
    topk, ids, bt = torch.zeros((n, k), **i32), torch.zeros(n, **i32), torch.zeros((2, 3), **i32)
    inst._resolve_residency(req_id_per_token=ids, block_table=bt, topk_indices=topk, block_size=4,
                            return_valid_counts=True, shared_rows=slice(0, n))
    written, req = torch.arange(n), torch.zeros(n + 2, **i32)
    inst.invalidate_written_slots(written, req)

    (kind_r, a_r, kw_r), (kind_i, a_i, kw_i) = calls
    assert (kind_r, kind_i) == ("r", "i")
    b = inspect.signature(REAL_RESOLVE).bind(*a_r, **kw_r).arguments
    assert b["host_cache"] is inst.host_cache and b["hot_cache"] is inst.hot.cache
    assert b["global_indices"] is topk and b["source_block_table"] is bt and b["source_block_size"] == 4
    assert b["device_global_indices"] is group.device_global_indices and b["lru_slots"] is group.lru_slots
    assert b["stats"] is group.swap_stats and b["lru_scratch"] is group._suffix_lru_scratch
    assert b["region_stride"] == hot and b["max_union_rows"] == 48 and b["attention_block_stride"] == 9
    assert b["valid_counts"].data_ptr() == shared.valid_topk_counts.data_ptr()
    assert b["resident_block_table"] is None and b["miss_mask"] is None
    bi = inspect.signature(REAL_INVALIDATE).bind(*a_i, **kw_i).arguments
    assert bi["device_global_indices"] is group.device_global_indices
    assert bi["req_id_per_token"].numel() == n and bi["written_slots"].numel() == n

    cfg = types.SimpleNamespace(max_union_rows=6 * 2048, device_buffer_size=7 * 2048)
    rt._check_residency_shared_memory(torch.device("cuda", 0), cfg)  # 79,140 B: fits
    with pytest.raises(ValueError, match="own HiSparse resolver"):
        rt._check_residency_shared_memory(
            torch.device("cuda", 0), types.SimpleNamespace(max_union_rows=16384, device_buffer_size=16384))


def _boot(tmp_path, env, rc):
    """Run sitecustomize with fake hisparse oracles that record themselves and exit rc."""
    pkg = tmp_path / "hisparse_mtp_patch"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text("def install_post_import_hook():\n    pass\n")
    for mod in ("split_oracle", "own_resolver"):
        (pkg / f"{mod}.py").write_text(
            f"open({str(tmp_path / 'ran')!r}, 'a').write({mod!r} + '\\n')\nraise SystemExit({rc})\n")
    base = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX_")}
    base.update(env, PYTHONPATH=f"{REPO}{os.pathsep}{tmp_path}",
                SUFFIX_BOOT_GATES_MARKER=str(tmp_path / "gates.claimed"))
    r = subprocess.run([sys.executable, "-c", "import sitecustomize"], env=base,
                       capture_output=True, text=True, cwd=tmp_path)
    ran = (tmp_path / "ran").read_text().split() if (tmp_path / "ran").exists() else []
    return r, ran


@pytest.mark.parametrize("own,rc", [(True, 0), (True, 1), (False, 1), (True, 3)])
def test_boot_runs_the_matching_oracle_and_fails_closed(tmp_path, own, rc):
    env = {"SUFFIX_SM120_HISPARSE_MTP": "1"}
    if own:
        env["SUFFIX_SM120_HISPARSE_OWN_RESOLVER"] = "1"
    r, ran = _boot(tmp_path, env, rc)
    which = "own resolver" if own else "split"
    assert ran == ["own_resolver" if own else "split_oracle"]
    assert f"[suffix sm120-hisparse-mtp] {which} oracle: exit {rc}" in r.stderr
    assert (r.returncode != 0) == (rc not in (0, 3)), r.stderr
    if rc == 1:
        assert f"residency {which} oracle did not pass" in r.stderr


def test_own_resolver_oracle_is_an_allowlisted_boot_gate():
    src = (REPO / "sitecustomize.py").read_text()
    g = {}
    exec(src[src.index("_BOOT_GATES = {"):src.index("\n_boot_gates = ")], g)
    assert g["_BOOT_GATES"]["hisparse_own_resolver_oracle"] == (
        ["-m", "hisparse_mtp_patch.own_resolver"], {})
