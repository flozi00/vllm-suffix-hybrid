# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash layout study: TP8 + EP8 versus DP attention (docs/dsv41-comms.md section 6).

ESTIMATES, CPU only. The model prices one decode step of a layout (tp, dp) with tp x dp = 8 GPUs, EP over all
8, against the TP8 + EP8 pool that serves today:

  comm     per layer: attention reduce (TP8: one all-reduce; TPxDP: sequence-parallel all-gather +
           reduce-scatter ~ one all-reduce over the TP group; DP8: none) + MoE (TP8: one all-reduce of the
           [tokens, 5120] output; DP>1: reduce-scatter of the padded [tokens_total, 5120] bf16 output + the
           all-gather dispatch of the NVFP4 activations). TP-group reduces use the host all-reduce curve
           (nccl_split.estimate_us "har"), DP/EP collectives NCCL: RS and AG = half of the best-NCCL
           all-reduce at the same bytes (ring: 7 hops of 14). `python -m suffix_hybrid.comm_layout bench`
           (boot gate dp_ag_rs_bench_dsv41) measures RS / AG / AR on silicon; pass its numbers to step().
  weights  dense bytes every rank streams per decode step: replicated tensors stay, TP-sharded ones grow by
           8 / tp (docs/dsv41-dense.md section 2: wq_b, wo_a, wo_b, shared expert, lm_head, Engram wkv).
  sync     DP lockstep: the CPU all-reduce that agrees the padded token count every step (DP_SYNC_US, a
           guess), dummy batches on idle ranks cost nothing extra (the step is the max over ranks) but
           every rank pads to the busiest one.

What it does NOT model (section 6 lists them): prefix-cache locality of the DP load balancer, prefill
(one rank carries a whole chunk), the FlashInfer h_q=64 specialization, MoE work on padded rows.

    python -m suffix_hybrid.comm_layout table          # CPU
    python -m suffix_hybrid.comm_layout bench          # 8 GPUs: NCCL RS / AG / AR + host AR at DP8 sizes
"""
from __future__ import annotations

import math
import os
import sys

from suffix_hybrid import nccl_split as ns

MARK = "[suffix comm-layout]"
HIDDEN, LAYERS, DRAFT_LAYERS = 5120, 40, 3
VERIFY_ROWS, DRAFT_ROWS = 6, 5            # per request at k = 5
GPUS = 8
BW_GBPS = 1597 * 0.85                     # GDDR7 effective, as in docs/dsv41-dense.md
PCIE_GBPS = 50.0
DISPATCH_BYTES = 2560 + 320 + 24 + 24     # NVFP4 activation + e4m3 scales + topk weights (f32) + ids (i32) per token
DP_SYNC_US = 200.0                        # CPU gloo all-reduce of the padded token counts per step (unmeasured)
BARRIER_US = 20.0

# MB streamed per rank per decode step at TP8, stock kernels (lane D inventory, ESTIMATE of the model's reads)
SHARDED_MB = {"wq_b": 232, "wo_a": 361, "wo_b": 232, "shared.gate_up": 131, "shared.down": 65,
              "lm_head": 331, "engram.wkv": 41}
REPLICATED_MB = {"fused_wqa_wkv": 407, "markov_w2": 331, "hc_fn": 169, "router": 157, "dspark.main_proj": 81,
                 "indexer.wq_b": 43, "compressor": 37, "indexer.weights_proj": 3}
WO_A_FP8_MB = 186                         # lane D's FP8 wo_a (stock reads a BF16 dequantization at load: 361)

# MB resident per rank at TP8 that a smaller TP replicates (43 = 40 target + 3 DSpark layers)
SHARED_RESIDENT_MB = 43 * (3.04 + 1.52)                          # shared expert gate_up + down
RESIDENT_SHARDED_MB = (43 * (5.41 + 8.39 + 5.41)                 # wq_b, wo_a, wo_b
                       + 165.5 + 165.5                           # lm_head, embedding (vocab-parallel)
                       + 2 * 20.28)                              # Engram wkv, 2 layers
KV_GIB_TP8 = 40.0                         # per-rank KV pool at --gpu-memory-utilization 0.93 (boot5: 37.17 at 0.90)
KV_TOKENS_PER_GIB = 10_867_561 / 37.17    # boot5.log:2472 "GPU KV cache size"
MODEL_GIB_TP8 = 40.66                     # boot5 "Model loading took"


def layouts() -> list[tuple[int, int]]:
    return [(tp, GPUS // tp) for tp in (8, 4, 2, 1)]


def name(tp: int, dp: int) -> str:
    return "TP8+EP8" if dp == 1 else f"TP{tp}xDP{dp}+EP8" if tp > 1 else "DP8+EP8"


def ar_host(nbytes: float) -> float:
    return ns.estimate_us("har", int(nbytes))


def ar_nccl(nbytes: float) -> float:
    return ns.estimate_us("nccl", int(nbytes))


def rs_nccl(nbytes: float) -> float:
    return 0.5 * ar_nccl(nbytes)


def ag_nccl(nbytes: float) -> float:
    return 0.5 * ar_nccl(nbytes)


def stream_mb(tp: int, dp: int | None = None, wo_a_fp8: bool = False) -> float:
    """Dense MB a rank reads per decode step. TP > 1 with DP > 1 turns on sequence parallelism, whose shared
    expert is replicated on the whole TP group (DeepseekV4MLP is_sequence_parallel -> disable_tp)."""
    dp = GPUS // tp if dp is None else dp
    sp_shared = ("shared.gate_up", "shared.down") if dp > 1 else ()
    total = sum(REPLICATED_MB.values())
    for k, mb in SHARDED_MB.items():
        mb = WO_A_FP8_MB if (k == "wo_a" and wo_a_fp8) else mb
        total += mb * (GPUS if k in sp_shared else GPUS / tp)
    return total


def extra_weights_gib(tp: int, dp: int | None = None) -> float:
    """Resident weights per rank beyond the TP8 layout (sequence parallelism replicates the shared expert)."""
    dp = GPUS // tp if dp is None else dp
    shared = SHARED_RESIDENT_MB * ((GPUS if dp > 1 else GPUS / tp) - 1)
    return (RESIDENT_SHARDED_MB * (GPUS / tp - 1) + shared) / 1024


def kv_capacity(tp: int, dp: int) -> tuple[float, float]:
    """(per-replica KV pool GiB, aggregate unique KV tokens): replicas hold different requests."""
    per = KV_GIB_TP8 - extra_weights_gib(tp, dp)
    return per, dp * per * KV_TOKENS_PER_GIB


def logits_us(tp: int, rows: int) -> float:
    """Vocab-parallel logits all-gather through the host buffer (floor + two barriers); none at tp = 1."""
    if tp == 1:
        return 0.0
    payload = rows * (129280 // tp) * 2
    return (tp - 1) * payload / (PCIE_GBPS * 1e3) + 2 * BARRIER_US


def layer_us(tp: int, dp: int, rows: int, costs=None) -> float:
    """Communication of one decoder layer; rows = tokens of ONE replica (already padded across replicas)."""
    ar, rs, ag = costs or (ar_host, rs_nccl, ag_nccl)
    s_rep = rows * HIDDEN * 2
    s_tot = dp * s_rep
    # TP8: one all-reduce. TPxDP (sequence parallel): host all-gather + NCCL reduce-scatter of the replica's rows.
    attn = 0.0 if tp == 1 else ar(s_rep) if dp == 1 else 0.5 * ar(s_rep) + (costs or (0, rs_nccl))[1](s_rep)
    moe = ar(s_rep) if dp == 1 else rs(s_tot) + ag(dp * rows * DISPATCH_BYTES)
    return attn + moe


def step(tp: int, dp: int, c: int, costs=None, wo_a_fp8: bool = False, small_gemm_x: float = 1.0) -> dict:
    """One decode step of c running requests (balanced across the DP replicas, k = 5), ms.

    small_gemm_x: how much slower than its byte floor the TP8 dense stream really runs (lane D: a 128x128
    CUTLASS tile gives 14 CTAs on 188 SMs for N <= 1792); the factor fades linearly to 1 at tp = 1, where the
    GEMMs are 8x wider (factor 1 + (x - 1) (tp - 1) / 7). 1.0 = every layout at its floor (the optimistic case for TP8, i.e. the worst for DP)."""
    per = math.ceil(c / dp)
    v, d = VERIFY_ROWS * per, DRAFT_ROWS * per
    ar = (costs or (ar_host,))[0]
    comm = LAYERS * layer_us(tp, dp, v, costs) + DRAFT_LAYERS * layer_us(tp, dp, d, costs)
    if tp > 1:
        comm += ar(v * HIDDEN * 2) + ar(d * HIDDEN * 2)                    # embedding reduces
        comm += logits_us(tp, v) + logits_us(tp, d)
        comm += 2 * 2 * BARRIER_US                                          # Engram head gathers (2 layers, target)
    if dp > 1:
        comm += DP_SYNC_US
    weights = stream_mb(tp, dp, wo_a_fp8) / BW_GBPS * (1 + (small_gemm_x - 1) * (tp - 1) / (GPUS - 1))  # MB / (MB/ms) = ms
    return {"comm_ms": comm / 1e3, "weights_ms": weights, "total_ms": comm / 1e3 + weights}


def delta_ms(tp: int, dp: int, c: int, costs=None, wo_a_fp8: bool = False, small_gemm_x: float = 1.0) -> float:
    """Step time of (tp, dp) minus TP8's, same costs (> 0 = slower)."""
    return (step(tp, dp, c, costs, wo_a_fp8, small_gemm_x)["total_ms"]
            - step(GPUS, 1, c, costs, False, small_gemm_x)["total_ms"])


def break_even(tp: int, dp: int, costs=None, wo_a_fp8: bool = False, cmax: int = 32,
               small_gemm_x: float = 1.0) -> int | None:
    """Smallest c <= cmax from which (tp, dp) is no slower than TP8 at every larger c, else None."""
    ok = [c for c in range(1, cmax + 1) if delta_ms(tp, dp, c, costs, wo_a_fp8, small_gemm_x) <= 0]
    return next((c for c in ok if all(x in ok for x in range(c, cmax + 1))), None)


def table(cs=(1, 8, 16, 32)) -> str:
    lines = ["ESTIMATE (model constants in suffix_hybrid/comm_layout.py), ms per decode step at k=5",
             f"{'layout':14s} {'dense MB':>9s} {'+weights GiB':>13s} {'KV GiB/replica':>15s} {'agg KV Mtok':>12s} "
             + " ".join(f"c={c:<2d} comm/weights/delta".rjust(26) for c in cs)]
    for tp, dp in layouts():
        kv, tok = kv_capacity(tp, dp)
        cells = []
        for c in cs:
            s = step(tp, dp, c)
            cells.append(f"{s['comm_ms']:6.1f} {s['weights_ms']:6.1f} {delta_ms(tp, dp, c):+7.1f}".rjust(26))
        lines.append(f"{name(tp, dp):14s} {stream_mb(tp, dp):9.0f} {extra_weights_gib(tp, dp):13.1f} {kv:15.1f} "
                     f"{tok / 1e6:12.1f} " + " ".join(cells))
    be = {name(tp, dp): break_even(tp, dp) for tp, dp in layouts()[1:]}
    lines.append(f"break-even c (no slower than TP8 from there up to 32): {be}")
    return "\n".join(lines)


# ------------------------------------------------------------------ silicon: NCCL AG / RS / AR at DP8 sizes
TOTAL_TOKENS = (48, 96, 192)   # padded tokens across 8 ranks at c = 8 / 16 / 32 (6 verify rows per request)


def _bench_worker(rank, world, port, q) -> None:
    import torch
    import torch.distributed as dist

    from suffix_hybrid.host_ar import HostAllReduce

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world, device_id=dev)
    cpu = dist.new_group(backend="gloo")
    out = {"rank": rank, "rows": [], "err": None}
    try:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        nccl = PyNcclCommunicator(group=cpu, device=dev)
        har = HostAllReduce(cpu, dev, oneshot_max_bytes=48 << 10, timeout_spins=1 << 26)
        for t in TOTAL_TOKENS:
            r = t // world
            x = torch.randn(t, HIDDEN, device=dev, dtype=torch.float32).to(torch.bfloat16)
            rs_out = torch.empty(r, HIDDEN, dtype=torch.bfloat16, device=dev)
            # the AgRs dispatch gathers four tensors per layer in one NCCL group (all2all.py AgRsAll2AllManager)
            parts = [torch.randint(0, 255, (r, 2560), dtype=torch.uint8, device=dev),
                     torch.randint(0, 255, (r, 320), dtype=torch.uint8, device=dev),
                     torch.randn(r, 6, device=dev), torch.randint(0, 384, (r, 6), dtype=torch.int32, device=dev)]
            outs = [torch.empty((t,) + p.shape[1:], dtype=p.dtype, device=dev) for p in parts]

            def ag():
                nccl.group_start()
                for o, p in zip(outs, parts):
                    nccl.all_gather(o, p)
                nccl.group_end()

            nccl.all_reduce(x)
            torch.cuda.synchronize()
            out["rows"].append({
                "t": t, "ar_nccl": ns.graph_time_us(lambda: nccl.all_reduce(x)),
                "ar_har": ns.graph_time_us(lambda: har.all_reduce(x)),
                "rs": ns.graph_time_us(lambda: nccl.reduce_scatter(rs_out, x)),
                "ag": ns.graph_time_us(ag)})
        har.destroy()
    except Exception:  # noqa: BLE001
        import traceback

        out["err"] = traceback.format_exc()[-1500:]
    q.put(out)
    try:
        dist.barrier(group=cpu)
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        pass


def bench() -> int:
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    if world != GPUS:
        print(f"{MARK} SKIP: {world} GPU(s)", flush=True)
        return 0
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29800 + os.getpid() % 1000
    procs = [ctx.Process(target=_bench_worker, args=(r, world, port, q)) for r in range(world)]
    [p.start() for p in procs]
    res = []
    try:
        for _ in procs:
            res.append(q.get(timeout=900))
    except Exception as exc:  # noqa: BLE001
        print(f"{MARK} FAIL: {len(res)}/{world} ranks reported ({exc!r})", flush=True)
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    errs = [r for r in res if r["err"]]
    for r in errs[:2]:
        print(f"{MARK} rank {r['rank']} ERROR: {r['err']}", flush=True)
    if errs or len(res) < world:
        return 1
    res.sort(key=lambda r: r["rank"])
    worst = [{k: max(r["rows"][i][k] for r in res) for k in res[0]["rows"][i]} for i in range(len(TOTAL_TOKENS))]
    print(f"{MARK} us per call, graphed, max over ranks; model = nccl_split estimate (RS, AG = 0.5 x best AR)",
          flush=True)
    for w in worst:
        t = w["t"]
        s_tot = t * HIDDEN * 2
        tp_layer, dp_layer = 2 * w["ar_har"], w["rs"] + w["ag"]
        print(f"{MARK} tokens {t:4d} ({s_tot >> 10} KiB): AR nccl {w['ar_nccl']:7.1f} (model {ar_nccl(s_tot):7.1f}) | "
              f"AR host {w['ar_har']:7.1f} (model {ar_host(s_tot):7.1f}) | RS {w['rs']:7.1f} "
              f"(model {rs_nccl(s_tot):7.1f}) | AG x4 {w['ag']:7.1f} (model "
              f"{ag_nccl(t * DISPATCH_BYTES):7.1f}) | layer: TP8 2xhost {tp_layer:7.1f} vs DP8 RS+AG "
              f"{dp_layer:7.1f} -> DP {'wins' if dp_layer < tp_layer else 'loses'} {abs(tp_layer - dp_layer):.1f} us "
              f"(x{LAYERS} = {abs(tp_layer - dp_layer) * LAYERS / 1e3:.1f} ms/step)", flush=True)
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["table", "bench"])
    a = ap.parse_args(argv)
    if a.cmd == "table":
        print(table())
        return 0
    return bench()


if __name__ == "__main__":
    sys.exit(main())
