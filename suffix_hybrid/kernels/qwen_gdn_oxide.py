# SPDX-License-Identifier: Apache-2.0
"""CPU (numpy) twin of the cuda-oxide K-GDN1 kernel
(kernels-oxide/kgdn1/src/main.rs, PTX entry kgdn1_decode_h16_hv48_k128).

Statement-for-statement transcription of the SIMT algorithm, vectorized over
the 32 lanes and 128 value rows: lane l owns K-slice {l, l+32, l+64, l+96};
per-lane partial dot products are summed in slice order, then reduced with the
same xor butterfly (16, 8, 4, 2, 1) the kernel uses, so the twin reproduces the
kernel's f32 reduction order (FMA contraction and ex2/lg2.approx aside).
bf16 inputs/outputs are raw uint16 bits exactly as the kernel reads/writes
them; f32 -> bf16 is round-to-nearest-even like the kernel's `f32_to_bf16`.

Arguments mirror the kernel params in interface.json order (pointers become
flat numpy arrays; strides are in elements). Runs on any host; the tests
check it against the torch reference of vLLM's semantics.
"""
from __future__ import annotations

import json
import os

import numpy as np

H, HV, K = 16, 48, 128
WARP_ROWS = 16
INTERFACE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "kernels-oxide", "kgdn1", "interface.json")


class Trap(RuntimeError):
    """The kernel's `debug::trap()` (slot >= slots)."""


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    b = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    nan = (b & 0x7FFFFFFF) > 0x7F800000
    return np.where(nan, np.uint16(0x7FC0), r)


def _butterfly(v):
    """v: [..., 32] per-lane values -> [..., 32] warp sums (xor order)."""
    lanes = np.arange(32)
    for m in (16, 8, 4, 2, 1):
        v = (v + v[..., lanes ^ m]).astype(np.float32)
    return v


def _lane_dot(a, b):
    """a, b: [..., 32, 4] lane slices -> per-lane partial, slice order."""
    p = a[..., 0] * b[..., 0]
    for j in (1, 2, 3):
        p = (p + a[..., j] * b[..., j]).astype(np.float32)
    return p.astype(np.float32)


def _slices(vec):
    """[K] -> [32 lanes, 4] with lane l holding {l, l+32, l+64, l+96}."""
    return np.asarray(vec, dtype=np.float32).reshape(4, 32).T


def _exp(x):
    return np.exp(np.float32(x), dtype=np.float32)


def kgdn1_twin(out, mixed_qkv, z, ba, a_log, dt_bias, norm_w, state, state_idx,
               T, slots, qkv_stride, z_stride, state_stride, act, scale, norm_eps):
    """Run the kernel grid (HV, T) on flat arrays. Mutates `out` (uint16) and
    `state` (float32) in place, like the kernel."""
    f32 = np.float32
    for t in range(T):
        for hv in range(HV):
            orow = (t * HV + hv) * K
            slot = int(state_idx[t])
            if slot <= 0:
                out[orow:orow + K] = 0
                continue
            if slot >= slots:
                raise Trap(f"slot {slot} >= slots {slots}")
            h = hv // (HV // H)
            row = t * qkv_stride
            q = _slices(bf16_to_f32(mixed_qkv[row + h * K: row + h * K + K]))
            k = _slices(bf16_to_f32(mixed_qkv[row + H * K + h * K: row + H * K + h * K + K]))
            qss = _butterfly(_lane_dot(q, q))[0]
            kss = _butterfly(_lane_dot(k, k))[0]
            q = (q * f32(f32(scale) / np.sqrt(f32(qss + f32(1e-6))))).astype(f32)
            k = (k * f32(f32(1.0) / np.sqrt(f32(kss + f32(1e-6))))).astype(f32)
            kq = _butterfly(_lane_dot(k, q))[0]
            brow = t * 2 * HV
            b = bf16_to_f32(ba[brow + hv])
            a = bf16_to_f32(ba[brow + HV + hv])
            x = f32(a + f32(dt_bias[hv]))
            sp = f32(np.log2(f32(1.0) + _exp(x)) * f32(np.log(2.0))) if x <= 20.0 else x
            decay = _exp(f32(-_exp(f32(a_log[hv])) * sp))
            beta = f32(1.0) / (f32(1.0) + _exp(-b))
            vrow = bf16_to_f32(mixed_qkv[row + 2 * H * K + hv * K: row + 2 * H * K + hv * K + K])
            base = slot * state_stride + hv * K * K
            S = state[base:base + K * K].reshape(K, K)            # [v, k]
            s = (S.reshape(K, 4, 32).transpose(0, 2, 1) * decay).astype(f32)  # [v, lane, j]
            pk = _butterfly(_lane_dot(s, k[None]))[:, 0]
            pq = _butterfly(_lane_dot(s, q[None]))[:, 0]
            dv = (beta * (vrow - pk)).astype(f32)
            s_new = (s + dv[:, None, None] * k[None]).astype(f32)
            state[base:base + K * K] = s_new.transpose(0, 2, 1).reshape(-1)
            o = (pq + dv * kq).astype(f32)
            ss = _butterfly(_lane_dot(_slices(o)[None], _slices(o)[None]))[0, 0]
            rstd = f32(1.0) / np.sqrt(f32(ss * f32(1.0 / K) + f32(norm_eps)))
            zv = bf16_to_f32(z[t * z_stride + hv * K: t * z_stride + hv * K + K])
            sg = (f32(1.0) / (f32(1.0) + np.exp(-zv, dtype=f32))).astype(f32)
            gate = zv * sg if act == 0 else sg
            y = (o * rstd * np.asarray(norm_w, dtype=f32) * gate).astype(f32)
            out[orow:orow + K] = f32_to_bf16(y)


def interface() -> dict:
    with open(INTERFACE) as f:
        return json.load(f)
