# SPDX-License-Identifier: Apache-2.0
"""Single-GPU silicon oracle + microbench for the SM120 NVFP4 DS-MLA kernels.

    python -m nvfp4_ds_mla_patch.oracle            # gates, exit 0 = PASS
    python -m nvfp4_ds_mla_patch.oracle --bench    # us/call vs stock fp8_ds_mla

Synthetic bf16 data at GLM 5.3 geometry (kv_lora 512, rope 64, topk 2048,
bmm1 scale 256^-0.5, per-rank heads 64/TP -> 8 at TP=8), no model needed.
Every gate is fatal (exit 1); exit 2 = could not run (no SM120 / imports).

(a) writer  our quant_store (through the patch helper) vs a torch reference
    of the 352 B row contract: bytes equal (e2m1 +-0 canonicalized),
    SF permutation, sf floor/saturation rows, -1 slots skipped, no stray
    writes, padded block-stride view; plus agreement with vLLM's fused
    Triton writer (fused_norm_rope, the non-HiSparse production writer).
(b) reader  decode vs (1) a torch dequant reference of the SAME cache:
    per-(token, head) rel-L2 <= 2e-2 — the kernel adds only f16 P / bf16
    partial+output rounding (~4e-3); (2) the stock fp8_ds_mla FlashInfer
    SM120 path on the same bf16 inputs: stock must match the SAME torch
    attention reference on its own 656 B rows within 5e-2 (FlashInfer runs
    e4m3 MMAs, so Q/P carry fp8 rounding) — this is the independent check
    that the reference semantics (scale, RoPE order, -1 masking) are vLLM's;
    and rel-L2(ours, stock) <= nvfp4ref~fp8ref (format noise, measured live)
    + stock~fp8ref + 0.02 (our kernel budget).
    T in {1, 6 (MTP k=5 verify), 32, 256 (prefill: ns=1)},
    HQ 8 (TP=8) and 64, -1 padding / holes / an all -1 token (exact 0),
    split-capacity merge (ns 32 / 6 / 1), padded block stride (bitwise),
    CUDA-graph replay with new indices (bitwise).
(c) --bench  graph-replayed us/call, ours vs stock, T in 1..64, HQ 8.
"""

import argparse
import json
import sys
import types

MARK = "[suffix sm120-nvfp4-ds-mla] ORACLE"
ROW = 352
FP8_ROW = 656
DIM, PE = 512, 64
TOPK = 2048
SCALE = 256 ** -0.5  # GLM 5.3 qk_head_dim 192 + 64
SF_PERM = [8 * (s & 3) + (s >> 2) for s in range(32)]
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------------------------------------------------------------------
# torch references (device-agnostic, CPU-tested in sm120/tests)
# ---------------------------------------------------------------------------
def e2m1_codes(x):
    """f32 -> e2m1 nibble: RNE (ties to the even code), satfinite at 6,
    sign bit 3 (cvt.rn.satfinite.e2m1x2.f32)."""
    import torch

    a = x.abs()
    c = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int()
         + (a >= 1.75).int() + (a > 2.5).int() + (a >= 3.5).int()
         + (a > 5.0).int())
    return (c | (torch.signbit(x).int() << 3)).to(torch.uint8)


def e4m3_bytes(x):
    """f32 -> e4m3fn byte, RNE + satfinite (cvt.rn.satfinite.e4m3x2.f32)."""
    import torch

    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)


def e4m3_float(b):
    import torch

    return b.contiguous().view(torch.float8_e4m3fn).float()


def quant_rows_ref(kv_c, k_pe):
    """bf16 [T, 512] / [T, 64] -> [T, 352] uint8 (the kernel recipe:
    sf = e4m3(max(amax * f32(1/6), 2^-9)); data = e2m1(x * (1 / sf)))."""
    import torch

    t = kv_c.shape[0]
    x = kv_c.float().view(t, 32, 16)
    sfv = torch.maximum(x.abs().amax(-1) * torch.tensor(1.0 / 6.0),
                        torch.tensor(2.0 ** -9))
    sf_b = e4m3_bytes(sfv)
    sf_f = e4m3_float(sf_b)
    inv = torch.where(sf_f == 0, torch.zeros_like(sf_f), 1.0 / sf_f)
    codes = e2m1_codes(x * inv[..., None]).view(t, 256, 2)
    rows = torch.empty(t, ROW, dtype=torch.uint8, device=kv_c.device)
    rows[:, :256] = codes[..., 0] | (codes[..., 1] << 4)
    rows[:, 256:320] = e4m3_bytes(k_pe.float())
    rows[:, [320 + p for p in SF_PERM]] = sf_b
    return rows


def canon(rows):
    """Map the e2m1 -0 nibble (8) to +0: numerically identical codes."""
    lo, hi = rows[..., :256] & 0xF, rows[..., :256] >> 4
    lo = lo.masked_fill(lo == 8, 0)
    hi = hi.masked_fill(hi == 8, 0)
    out = rows.clone()
    out[..., :256] = lo | (hi << 4)
    return out


def dequant_rows_ref(rows):
    """[N, 352] uint8 -> (latent [N, 512], rope [N, 64]) f32."""
    import torch

    n = rows.shape[0]
    lut = torch.tensor(E2M1, device=rows.device)
    data = rows[:, :256].long()
    nib = torch.stack((data & 0xF, data >> 4), -1).view(n, DIM)
    sf = e4m3_float(rows[:, [320 + p for p in SF_PERM]])
    lat = lut[nib] * sf.repeat_interleave(16, -1)
    return lat, e4m3_float(rows[:, 256:320])


def dequant_fp8_rows(rows):
    """[N, 656] fp8_ds_mla rows: [0,512) e4m3, [512,528) 4 x f32 tile
    scales, [528,656) bf16 RoPE -> (latent [N, 512], rope [N, 64]) f32."""
    import torch

    scales = rows[:, 512:528].contiguous().view(torch.float32)
    lat = e4m3_float(rows[:, :512]) * scales.repeat_interleave(128, -1)
    rope = rows[:, 528:656].contiguous().view(torch.bfloat16).float()
    return lat, rope


def attention_ref(q, lat, rope, topk, scale):
    """q [T, H, 576] f32, rows by slot id, topk [T, C] (-1 = masked) ->
    [T, H, 512] f32; an all-masked token is 0."""
    import torch

    out = torch.zeros(q.shape[0], q.shape[1], DIM, device=q.device)
    for t in range(q.shape[0]):
        idx = topk[t][topk[t] >= 0].long()
        if idx.numel() == 0:
            continue
        k = torch.cat([lat[idx], rope[idx]], -1)
        p = torch.softmax((q[t] @ k.T) * scale, -1)
        out[t] = p @ lat[idx]
    return out


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def row_rel_l2(a, b, keep):
    """max over kept (token, head) rows of rel-L2."""
    d = (a.float() - b.float()).norm(dim=-1)
    n = b.float().norm(dim=-1).clamp_min(1e-30)
    return float((d / n)[keep].max())


# ---------------------------------------------------------------------------
# synthetic GLM-shaped data
# ---------------------------------------------------------------------------
def randn(gen, dev, *shape):
    import torch

    return torch.randn(*shape, generator=gen, device=gen.device).to(dev)


def randperm(gen, dev, n):
    import torch

    return torch.randperm(n, generator=gen, device=gen.device).to(dev)


def make_kv(n, dev, gen, special=True):
    import torch

    chan = torch.exp(randn(gen, dev, DIM) * 0.75)
    kv_c = (randn(gen, dev, n, DIM) * chan).bfloat16()
    k_pe = (randn(gen, dev, n, PE) * 2.0).bfloat16()
    if not special:
        return kv_c, k_pe
    kv_c[0] = 0                     # sf floor 2^-9 row
    kv_c[1, :16] = 5000.0           # saturating block (sf 448)
    kv_c[2] = (kv_c[2].float() * 1e-4).bfloat16()  # tiny row
    return kv_c, k_pe


def make_topk(t, nslots, dev, gen, full_first=True):
    import torch

    topk = torch.full((t, TOPK), -1, dtype=torch.int32, device=dev)
    for i in range(t):
        n = TOPK if (i == 0 and full_first) else (
            0 if i == 1 else int(randperm(gen, "cpu", TOPK)[0]) + 1)
        topk[i, :n] = randperm(gen, dev, nslots)[:n].int()
        if i % 2:  # -1 holes anywhere, not just a tail
            topk[i] = topk[i, randperm(gen, dev, TOPK)]
    return topk


class Ours:
    """The patch's own helpers on a fake impl (the path vLLM runs)."""

    def __init__(self, dev):
        import torch

        import nvfp4_ds_mla_patch as P
        from suffix_hybrid import oxide_kernels

        P._native()
        oxide_kernels.ensure_loaded(P.FAMILY, dev)
        self.P = P
        self.impl = types.SimpleNamespace(
            scale=SCALE,
            _nvfp4_num_sms=torch.cuda.get_device_properties(dev).multi_processor_count)

    def write(self, kv_c, k_pe, cache, slots):
        self.P._suffix_nvfp4_ds_mla_write(kv_c, k_pe.unsqueeze(1), cache, slots)

    def decode(self, q, cache, topk, out=None):
        if out is None:
            out = q.new_empty(q.shape[0], q.shape[1], DIM)
        return self.P._suffix_nvfp4_ds_mla_decode(self.impl, q, cache, topk, out)


class Stock:
    """vLLM 0.30.0 fp8_ds_mla on SM120: C++ writer + FlashInfer sparse MLA."""

    def __init__(self, dev):
        from vllm import _custom_ops as ops
        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla as fi)
        from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
            _get_workspace_buffer)

        self.ops, self.fi, self.ws = ops, fi, _get_workspace_buffer(dev)

    def write(self, kv_c, k_pe, cache, slots):
        import torch

        self.ops.concat_and_cache_mla(kv_c, k_pe, cache, slots, "fp8_ds_mla",
                                      torch.ones(1, device=kv_c.device))

    def decode(self, q, cache, topk, out=None):
        if out is None:
            out = q.new_empty(q.shape[0], q.shape[1], DIM)
        self.fi(query=q.unsqueeze(1), kv_cache=cache.unsqueeze(1),
                workspace_buffer=self.ws, qk_nope_head_dim=192,
                kv_lora_rank=DIM, qk_rope_head_dim=PE,
                block_tables=topk.unsqueeze(1), seq_lens=None,
                max_seq_len=topk.shape[1], out=out.unsqueeze(1),
                bmm1_scale=SCALE, bmm2_scale=1.0,
                sparse_mla_top_k=topk.shape[1], kv_scale_format="arbitrary_fp32")
        return out


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def gate_writer(ours, dev, gen, report):
    import torch

    bs, nb, t = 64, 16, 257
    nslots = bs * nb
    kv_c, k_pe = make_kv(t, dev, gen)
    slots = randperm(gen, dev, nslots)[:t]
    slots[3 + randperm(gen, dev, t - 3)[:20]] = -1  # rows 0-2 are special
    cache = torch.full((nb, bs, ROW), 0xA5, dtype=torch.uint8, device=dev)
    ours.write(kv_c, k_pe, cache, slots)
    torch.cuda.synchronize()
    ref = quant_rows_ref(kv_c, k_pe)
    flat = cache.view(-1, ROW)
    live = slots >= 0
    got = canon(flat[slots[live]])
    bad = int((got != canon(ref[live])).any(-1).sum())
    untouched = torch.ones(nslots, dtype=torch.bool, device=dev)
    untouched[slots[live]] = False
    stray = int((flat[untouched] != 0xA5).any(-1).sum())
    sat = int(ref[1, 320 + SF_PERM[0]])
    report("writer_bytes_vs_ref", bad == 0 and stray == 0 and sat == 0x7E,
           f"{bad}/{int(live.sum())} rows differ, {stray} stray rows, "
           f"saturated sf byte {sat:#x}")

    # padded block stride (HiSparse hot-view geometry)
    pitch = bs * ROW + 64
    raw = torch.full((nb * pitch,), 0xA5, dtype=torch.uint8, device=dev)
    view = raw.as_strided((nb, bs, ROW), (pitch, ROW, 1))
    ours.write(kv_c, k_pe, view, slots)
    torch.cuda.synchronize()
    same = torch.equal(view.reshape(-1, ROW)[slots[live]], flat[slots[live]])
    pads = int((raw.view(nb, pitch)[:, bs * ROW:] != 0xA5).sum())
    report("writer_padded_block_stride", same and pads == 0,
           f"rows equal {same}, {pads} pad bytes touched")

    # vLLM's fused Triton writer (non-HiSparse production path)
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    n = 128
    kv_in = randn(gen, dev, n, DIM).bfloat16()
    kpe_in = randn(gen, dev, n, PE).bfloat16()
    ang = randn(gen, dev, n, 32) * 3.0
    cos_sin = torch.cat([ang.cos(), ang.sin()], -1).float()
    tri = torch.zeros((nb, bs, ROW), dtype=torch.uint8, device=dev)
    kv_out, kpe_out = torch.empty_like(kv_in), torch.empty_like(kpe_in)
    sl = torch.arange(n, device=dev)
    fused_norm_rope(
        torch.arange(n, device=dev), torch.randn(n, 256, device=dev).bfloat16(),
        torch.ones(256, device=dev).bfloat16(), 1e-6, kv_in,
        torch.ones(DIM, device=dev).bfloat16(), 1e-6, kpe_in, cos_sin,
        None, None, None, 1e-6, None,
        torch.zeros(n, TOPK, dtype=torch.int32, device=dev),
        slot_mapping=sl, mla_kv_cache=tri, mla_kv_cache_dtype="nvfp4_ds_mla",
        has_indexer=False, kv_c_out=kv_out, k_pe_out=kpe_out)
    mine = torch.zeros_like(tri)
    ours.write(kv_out, kpe_out, mine, sl)
    torch.cuda.synchronize()
    a, b = tri.view(-1, ROW)[:n], mine.view(-1, ROW)[:n]
    frac = float((canon(a) != canon(b)).float().mean())
    la, ra = dequant_rows_ref(a)
    lb, rb = dequant_rows_ref(b)
    err = max(rel_l2(lb, la), rel_l2(rb, ra))
    # Triton quantizes the f32 normed kv_c, we get its bf16 copy: rare
    # rounding-boundary flips only.
    report("writer_vs_vllm_triton", frac <= 0.02 and err <= 0.02,
           f"{frac:.4%} bytes differ, dequant rel-L2 {err:.2e}")


def gate_reader(ours, stock, dev, gen, report):
    import torch

    bs, nb = 64, 256
    nslots = bs * nb
    # no saturating rows here: 5000 clips to 6 * 448 in nvfp4 (writer gate
    # covers it) and would dominate the format-noise comparison with stock
    kv_c, k_pe = make_kv(nslots, dev, gen, special=False)
    all_slots = torch.arange(nslots, device=dev)
    cache = torch.zeros((nb, bs, ROW), dtype=torch.uint8, device=dev)
    ours.write(kv_c, k_pe, cache, all_slots)
    fp8 = torch.zeros((nb, bs, FP8_ROW), dtype=torch.uint8, device=dev)
    stock.write(kv_c, k_pe, fp8, all_slots)
    torch.cuda.synchronize()
    lat, rope = dequant_rows_ref(cache.view(-1, ROW))
    flat, frope = dequant_fp8_rows(fp8.view(-1, FP8_ROW))
    for t, h in [(1, 8), (6, 8), (32, 8), (1, 64), (6, 64), (256, 8)]:
        q = randn(gen, dev, t, h, DIM + PE).bfloat16()
        topk = make_topk(t, nslots, dev, gen)
        out = ours.decode(q, cache, topk)
        torch.cuda.synchronize()
        ref = attention_ref(q.float(), lat, rope, topk, SCALE)
        valid = (topk >= 0).any(-1)
        keep = valid[:, None].expand(t, h)
        zero_ok = bool((out[~valid] == 0).all())
        e_row = row_rel_l2(out, ref, keep)
        report(f"reader_vs_ref_T{t}_H{h}", e_row <= 2e-2 and zero_ok
               and bool(torch.isfinite(out).all()),
               f"max row rel-L2 {e_row:.2e}, all-masked token zero {zero_ok}")
        if t > 64:
            continue  # stock decode kernel is <= 64 tokens (prefill path)
        st = stock.decode(q, fp8, topk)
        torch.cuda.synchronize()
        fref = attention_ref(q.float(), flat, frope, topk, SCALE)
        v = valid
        e_stock_ref = rel_l2(st[v], fref[v])
        e_fmt = rel_l2(ref[v], fref[v])
        e_ours_stock = rel_l2(out[v], st[v])
        bound = e_fmt + e_stock_ref + 0.02
        report(f"reader_vs_stock_T{t}_H{h}",
               e_stock_ref <= 0.05 and e_ours_stock <= bound,
               f"stock~fp8ref {e_stock_ref:.2e} (<= 5e-2: reference semantics "
               f"== FlashInfer's); ours~stock {e_ours_stock:.3e} <= {bound:.3e} "
               f"(format noise nvfp4ref~fp8ref {e_fmt:.3e})")

    # padded block stride: bitwise equal to the dense cache
    pitch = bs * ROW + 128
    raw = torch.zeros((nb * pitch,), dtype=torch.uint8, device=dev)
    view = raw.as_strided((nb, bs, ROW), (pitch, ROW, 1))
    view.copy_(cache)
    q = randn(gen, dev, 6, 8, DIM + PE).bfloat16()
    topk = make_topk(6, nslots, dev, gen)
    a, b = ours.decode(q, cache, topk), ours.decode(q, view, topk)
    torch.cuda.synchronize()
    report("reader_padded_block_stride", torch.equal(a, b), "bitwise vs dense")

    # CUDA graph: capture once, replay with new indices == eager
    out = torch.empty(6, 8, DIM, dtype=torch.bfloat16, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ours.decode(q, cache, topk, out)  # warm (allocator)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ours.decode(q, cache, topk, out)
    topk.copy_(make_topk(6, nslots, dev, gen))
    g.replay()
    eager = ours.decode(q, cache, topk)
    torch.cuda.synchronize()
    report("reader_cuda_graph_replay", torch.equal(out, eager),
           "graph replay with new indices bitwise == eager")


def bench(ours, stock, dev, args):
    import torch

    gen = torch.Generator(device=dev).manual_seed(1)
    bs, nb = 64, args.bench_slots // 64
    nslots = bs * nb
    cache = torch.zeros((nb, bs, ROW), dtype=torch.uint8, device=dev)
    fp8 = torch.zeros((nb, bs, FP8_ROW), dtype=torch.uint8, device=dev)
    for lo in range(0, nslots, 1 << 16):  # fill in chunks (writer temps)
        kv_c, k_pe = make_kv(min(1 << 16, nslots - lo), dev, gen)
        sl = torch.arange(lo, lo + kv_c.shape[0], device=dev)
        ours.write(kv_c, k_pe, cache, sl)
        stock.write(kv_c, k_pe, fp8, sl)
    rows = []
    for t in args.bench_tokens:
        q = randn(gen, dev, t, args.heads, DIM + PE).bfloat16()
        topk = torch.stack([randperm(gen, dev, nslots)[:TOPK] for _ in range(t)]).int()
        res = {"T": t, "HQ": args.heads, "topk": TOPK}
        for name, impl, c in (("ours_us", ours, cache), ("stock_us", stock, fp8)):
            out = torch.empty(t, args.heads, DIM, dtype=torch.bfloat16, device=dev)
            impl.decode(q, c, topk, out)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    impl.decode(q, c, topk, out)
            g.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(args.iters):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            res[name] = round(e0.elapsed_time(e1) * 1e3 / (10 * args.iters), 2)
        res["speedup"] = round(res["stock_us"] / res["ours_us"], 3)
        rows.append(res)
        print(f"{MARK} bench T={t:3d} HQ={args.heads} topk={TOPK}: ours "
              f"{res['ours_us']:8.2f} us  stock fp8_ds_mla {res['stock_us']:8.2f} us"
              f"  speedup {res['speedup']:.2f}x", flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", action="store_true", help="microbench only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--heads", type=int, default=8, help="bench per-rank heads")
    ap.add_argument("--bench-tokens", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 2, 4, 6, 8, 16, 32, 64])
    ap.add_argument("--bench-slots", type=int, default=1 << 20)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args(argv)
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
            print(f"{MARK} NOT RUN: needs an SM120 (cc 12.x) GPU", flush=True)
            return 2
        dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(dev)
        ours, stock = Ours(dev.index), Stock(dev)
    except Exception as exc:  # noqa: BLE001 - the verdict is the evidence
        print(f"{MARK} NOT RUN: {type(exc).__name__}: {exc}", flush=True)
        return 2
    if args.bench:
        rows = bench(ours, stock, dev, args)
        if args.json:
            print(json.dumps({"bench": rows}), flush=True)
        return 0
    results = []

    def report(name, ok, detail):
        results.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"{MARK} {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    gen = torch.Generator().manual_seed(0)  # CPU: reproducible across GPUs
    for gate in (gate_writer, gate_reader):
        try:
            gate(ours, dev, gen, report) if gate is gate_writer else gate(
                ours, stock, dev, gen, report)
        except Exception as exc:  # noqa: BLE001
            report(gate.__name__, False, f"raised {type(exc).__name__}: {exc}")
    ok = all(r["pass"] for r in results)
    if args.json:
        print(json.dumps({"pass": ok, "results": results}), flush=True)
    print(f"{MARK} NVFP4-DSMLA ORACLE {'PASS' if ok else 'FAIL'} "
          f"({sum(r['pass'] for r in results)}/{len(results)} gates)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
