# SPDX-License-Identifier: Apache-2.0
"""On-silicon numerics oracle for the SM120 NVFP4-KV fa2 route.

Gate (c) of dossiers/gemma-hd512-nvfp4.md: must pass on the real GPU before
any pool boots ``--kv-cache-dtype nvfp4`` at a head_dim the route has not yet
been proven at (first target: gemma-4 full-attn head_dim=512).

What it exercises is exactly the production byte path, nothing re-implemented:
  store  = vLLM's own ``_C_cache_ops.reshape_and_cache_flash(..., "nvfp4")``
           into a (B, 2H, N, full_dim) HND cache (FlashInferImpl layout),
  views  = vLLM's ``nvfp4_split_data_scale`` (the forward() split),
  kernel = FlashInfer ``fa2`` Batch{Prefill,Decode} wrappers with
           kv_data_type=uint8 + kv_cache_sf (patch H6/H7/H9 route), JIT'd with
           the V-SF de-swizzle define (patch ensure_deswizzle_flag()).
The reference dequantizes the cache BYTES in torch (e2m1 LUT x e4m3 SF x
global scale; K SF linear, V SF 4-token swizzled per the store kernel) and runs
fp32 attention — independent of the kernel under test. Gates:
  * decoder sanity: dequant(cache) vs the bf16 input  (store/decoder bug)
  * kernel vs dequant reference, tight                 (kernel/SF/layout bug)
  * kernel vs bf16-input attention, loose               (end-to-end sanity)

Run in the pod/CI image (PYTHONPATH=/plugins):
    python -m nvfp4_kv_patch.oracle [--shapes 512:16:2,256:16:8] [--json]
Exit 0 = PASS, 1 = FAIL, 2 = could not run (no SM120 / import error).
"""

import argparse
import json
import math
import sys

from . import SF_VEC_SIZE, ensure_deswizzle_flag, is_sm120

PAGE = 64
# e2m1 code -> value (bit 3 = sign).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# Gates. The kernel accumulates in fp32 over bf16-rounded P, so vs an exact
# fp32 reference over the SAME dequantized bytes it must be near-exact.
COS_KERNEL_MIN = 0.9995
REL_KERNEL_MAX = 2e-2
COS_DECODE_MIN = 0.98     # nvfp4 round-trip of gaussian data is ~0.99
COS_E2E_MIN = 0.95        # attention output vs un-quantized inputs


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _rel(a, b):
    return float((a.float() - b.float()).abs().max()
                 / b.float().abs().max().clamp_min(1e-30))


def dequant_side(data, sf, swizzled, global_scale):
    """(B,H,N,D/2) uint8 + (B,H,N,D/16) e4m3 views -> (B,H,N,D) fp32."""
    import torch

    lut = torch.tensor(_E2M1, device=data.device)
    d = data.to(torch.int64)
    vals = torch.stack((lut[d & 0xF], lut[d >> 4]), dim=-1).flatten(-2)
    s = sf.float()
    if swizzled:
        # Store wrote logical (t, g) at swizzle_scale_offset(t, g, S) in the
        # per-(page, head) [N*S] region; gather it back.
        n, g = s.shape[2], s.shape[3]
        t = torch.arange(n, device=s.device).view(n, 1)
        c = torch.arange(g, device=s.device).view(1, g)
        grp = g // 4
        flat = ((t // 4) * 4 + c // grp) * g + (c % grp) * 4 + t % 4
        s = s.flatten(2)[:, :, flat.flatten()].view(s.shape)
    return vals * s.repeat_interleave(SF_VEC_SIZE, dim=-1) * global_scale


def _ref_attn(q, k, v, sm_scale):
    """q (Lq,Hq,D), k/v (L,Hkv,D) fp32, causal with q at the tail."""
    import torch

    lq, hq, _ = q.shape
    lk, hkv, _ = k.shape
    k = k.repeat_interleave(hq // hkv, dim=1)
    v = v.repeat_interleave(hq // hkv, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k) * sm_scale
    pos_q = torch.arange(lk - lq, lk, device=q.device).view(lq, 1)
    mask = torch.arange(lk, device=q.device).view(1, lk) > pos_q
    logits.masked_fill_(mask, float("-inf"))
    return torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)


def _make_kv(n_tok, hkv, d, pattern, gen):
    import torch

    x = torch.randn(n_tok, hkv, d, generator=gen, device="cuda")
    if pattern == "adversarial":
        # Per 16-element block magnitudes over 1e-4..1e2 (e4m3 SF subnormals
        # through large), plus all-zero blocks (SF=0 path).
        g = x.view(n_tok, hkv, d // SF_VEC_SIZE, SF_VEC_SIZE)
        mag = 10 ** (torch.rand(g.shape[:-1], generator=gen, device="cuda")
                     * 6 - 4)
        mag[torch.rand(mag.shape, generator=gen, device="cuda") < 0.05] = 0
        x = (g * mag.unsqueeze(-1)).view(n_tok, hkv, d)
    return x.to(torch.bfloat16)


def run_case(shape, pattern, kv_lens, q_len, scales, wrapper_kind, seed=0):
    import torch
    from vllm.utils.torch_utils import (nvfp4_kv_cache_full_dim,
                                        nvfp4_split_data_scale)

    import flashinfer

    d, hq, hkv = shape
    k_scale, v_scale = scales
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages_per = [math.ceil(n / PAGE) for n in kv_lens]
    num_pages = sum(pages_per) + 3  # spare pages stay garbage-free zeros
    page_ids = torch.randperm(num_pages, generator=gen, device="cuda")

    cache = torch.zeros(num_pages, 2 * hkv, PAGE, nvfp4_kv_cache_full_dim(d),
                        dtype=torch.uint8, device="cuda")
    keys, vals, slots, indices, indptr, last = [], [], [], [], [0], []
    cur = 0
    for n, p in zip(kv_lens, pages_per):
        ids = page_ids[cur:cur + p]
        cur += p
        pos = torch.arange(n, device="cuda")
        slots.append(ids[pos // PAGE] * PAGE + pos % PAGE)
        keys.append(_make_kv(n, hkv, d, pattern, gen))
        vals.append(_make_kv(n, hkv, d, pattern, gen))
        indices.append(ids)
        indptr.append(indptr[-1] + p)
        last.append(n - (p - 1) * PAGE)

    # --- production store (FlashInferImpl.do_kv_cache_update, nvfp4 branch)
    k_cache, v_cache = cache.transpose(1, 2).split(hkv, dim=-2)
    ks = torch.tensor([k_scale], dtype=torch.float32, device="cuda")
    vs = torch.tensor([v_scale], dtype=torch.float32, device="cuda")
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        torch.cat(keys), torch.cat(vals), k_cache, v_cache,
        torch.cat(slots).to(torch.int64), "nvfp4", ks, vs)

    # --- production read views (FlashInferImpl.forward, HND identity order)
    k_side, v_side = cache.split(hkv, dim=1)
    k_data, k_sf = nvfp4_split_data_scale(k_side)
    v_data, v_sf = nvfp4_split_data_scale(v_side)

    k_deq = dequant_side(k_data, k_sf, False, k_scale)  # (B,H,N,D)
    v_deq = dequant_side(v_data, v_sf, True, v_scale)

    sm_scale = 1.0 / math.sqrt(d)
    qs = [torch.randn(q_len, hq, d, generator=gen, device="cuda")
          .to(torch.bfloat16) for _ in kv_lens]
    q = torch.cat(qs)
    ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor(indptr, **i32)
    kv_indices = torch.cat(indices).to(torch.int32)
    kv_last = torch.tensor(last, **i32)
    dtypes = dict(q_data_type=torch.bfloat16, kv_data_type=torch.uint8,
                  o_data_type=torch.bfloat16)
    if wrapper_kind == "decode":
        assert q_len == 1
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "HND", use_tensor_cores=True, backend="fa2")
        w.plan(kv_indptr, kv_indices, kv_last, hq, hkv, d, PAGE,
               pos_encoding_mode="NONE", sm_scale=sm_scale, **dtypes)
    else:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "HND", backend="fa2")
        qo_indptr = torch.arange(len(kv_lens) + 1, **i32) * q_len
        w.plan(qo_indptr, kv_indptr, kv_indices, kv_last, hq, hkv, d, PAGE,
               causal=True, sm_scale=sm_scale, **dtypes)
    out = w.run(q, (k_data, v_data), k_scale=k_scale, v_scale=v_scale,
                kv_cache_sf=(k_sf, v_sf))
    torch.cuda.synchronize()

    worst = {"cos_kernel": 1.0, "rel_kernel": 0.0, "cos_e2e": 1.0,
             "cos_decode": 1.0}
    for i, (n, ids) in enumerate(zip(kv_lens, indices)):
        # Gather through the page table (proves paging math, not just data).
        kd = k_deq[ids].permute(0, 2, 1, 3).reshape(-1, hkv, d)[:n]
        vd = v_deq[ids].permute(0, 2, 1, 3).reshape(-1, hkv, d)[:n]
        o = out[i * q_len:(i + 1) * q_len]
        ref = _ref_attn(qs[i], kd, vd, sm_scale)
        e2e = _ref_attn(qs[i], keys[i].float(), vals[i].float(),
                        sm_scale)
        worst["cos_kernel"] = min(worst["cos_kernel"], _cos(o, ref))
        worst["rel_kernel"] = max(worst["rel_kernel"], _rel(o, ref))
        worst["cos_e2e"] = min(worst["cos_e2e"], _cos(o, e2e))
        worst["cos_decode"] = min(worst["cos_decode"],
                                  _cos(kd, keys[i]), _cos(vd, vals[i]))
    # e2e/decode gates only mean something on well-scaled gaussian data; the
    # adversarial pattern deliberately loses precision in tiny blocks.
    strict = pattern == "gauss"
    worst["pass"] = (
        worst["cos_kernel"] >= COS_KERNEL_MIN
        and worst["rel_kernel"] <= REL_KERNEL_MAX
        and (not strict or worst["cos_decode"] >= COS_DECODE_MIN)
        and (not strict or worst["cos_e2e"] >= COS_E2E_MIN))
    return worst


# (name, kv_lens, q_len, wrapper). kv lens hit: 1 token, exact page multiple,
# ragged tails, > 4 pages, split-KV scale (>=256-token chunks).
CASES = (
    ("decode", (1, 64, 65, 300, 777, 1024, 2049, 4100), 1, "decode"),
    ("spec_verify_k8", (9, 130, 700, 3001), 9, "prefill"),
    ("prefill", (1500,), 1500, "prefill"),
    ("prefix_cached", (1200, 2500), 300, "prefill"),
)
PATTERNS = (("gauss", (1.0, 1.0)), ("gauss", (0.5, 2.0)),
            ("adversarial", (1.0, 1.0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shapes", default="512:16:2,256:16:8",
                    help="head_dim:num_q_heads:num_kv_heads,... "
                         "(default: gemma-4 full-attn, then SWA)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not is_sm120():
        print("oracle: not an SM120 (cc 12.x) device; nothing to verify",
              file=sys.stderr)
        return 2
    ensure_deswizzle_flag()  # before flashinfer JITs anything
    try:
        import flashinfer  # noqa: F401
        import vllm._custom_ops  # noqa: F401  (registers _C_cache_ops)
    except Exception as exc:
        print(f"oracle: import failed: {exc}", file=sys.stderr)
        return 2

    ok = True
    for spec in args.shapes.split(","):
        shape = tuple(int(x) for x in spec.split(":"))
        for name, kv_lens, q_len, kind in CASES:
            for pattern, scales in PATTERNS:
                row = {"head_dim": shape[0], "hq": shape[1], "hkv": shape[2],
                       "case": name, "pattern": pattern, "scales": scales}
                try:
                    row.update(run_case(shape, pattern, kv_lens, q_len,
                                        scales, kind))
                except Exception as exc:  # JIT/smem/plan failure = verdict
                    row.update({"pass": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                ok &= row["pass"]
                if args.json:
                    print(json.dumps(row), flush=True)
                else:
                    tag = "PASS" if row["pass"] else "FAIL"
                    detail = row.get("error") or " ".join(
                        f"{k}={row[k]:.5f}" for k in
                        ("cos_kernel", "rel_kernel", "cos_decode", "cos_e2e"))
                    print(f"{tag} hd={shape[0]} {name:<15} {pattern:<11} "
                          f"ks/vs={scales} {detail}", flush=True)
    print(f"oracle verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
