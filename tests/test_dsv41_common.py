# SPDX-License-Identifier: Apache-2.0
"""dsv41_common: the MXFP8 quantiser spec (vs an independent definition), the F8_128x4 swizzle,
and the capture/replay boot-gate logic (fake graph runner on CPU)."""
import math

import pytest

torch = pytest.importorskip("torch")
from suffix_hybrid.helion_kernels import dsv41_common as cm  # noqa: E402


def independent_sf(amax: float) -> int:
    """FlashInfer's UE8M0 scale of a group with this f32 amax, from the definition:
    the smallest power of two >= amax / 448 (normalised value computed in f32)."""
    norm = float(torch.tensor(amax, dtype=torch.float32) * torch.tensor(1.0 / 448.0, dtype=torch.float32))
    if norm <= 0:
        return 0
    m, e = math.frexp(norm)             # norm = m * 2^e, m in [0.5, 1)
    exp_unbiased = e - 1 if m == 0.5 else e          # exact power of two keeps its exponent
    return max(0, min(254, exp_unbiased + 127))


def test_quant_reference_matches_the_definition_on_normal_groups():
    g = torch.Generator().manual_seed(1)
    y = (torch.randn(40, 160, generator=g) * torch.logspace(-3, 3, 40)[:, None]).to(torch.bfloat16)
    q, scales = cm.mxfp8_quant_reference(y)
    yf = y.float().reshape(40, 5, 32)
    deq = cm.mxfp8_dequant(q, scales).reshape(40, 5, 32)
    rows = torch.arange(40)
    for gi in range(5):
        off = ((rows // 128) * 2 + gi // 4) * 512 + (rows % 32) * 16 + ((rows % 128) // 32) * 4 + gi % 4
        for r in range(40):
            amax = float(yf[r, gi].abs().max())
            assert int(scales[off[r]]) == independent_sf(amax), (r, gi, amax)
    scale_val = 2.0 ** (torch.tensor([independent_sf(float(a)) for a in yf.abs().amax(-1).flatten()]).float() - 127)
    err = (deq - yf).abs()
    bound = 0.0625 * yf.abs() + scale_val.reshape(40, 5, 1) * 2.0 ** -9
    assert bool((err <= bound).all())
    assert float(q.float().abs().max()) <= 448.0 and bool(q.float().isfinite().all())


def test_quant_helper_equals_reference_incl_edge_groups():
    """the traced helper (pure torch) == the transcription, on groups that hit every branch:
    zero, f32-subnormal and bf16-subnormal-normalised amax, exact powers of two, 448, huge."""
    amax_values = [0.0, 1e-45, 1e-40, 1e-38, 5.9e-39, 1e-30, 2.0 ** -126, 1.0, 448.0, 447.99, 449.0,
                   256.0, 1e30, 3e38, 7.0e-3]
    rows = []
    for a in amax_values:
        row = torch.zeros(32)
        row[3] = a
        row[17] = -a / 3.0
        rows.append(row)
    y = torch.stack(rows).to(torch.bfloat16)
    q_h, sf_h = cm.mxfp8_group_quant(y.float().reshape(len(rows), 1, 32))
    q_r, scales = cm.mxfp8_quant_reference(y)
    assert torch.equal(q_h.reshape(len(rows), 32).view(torch.uint8), q_r.view(torch.uint8))
    r = torch.arange(len(rows))
    got_sf = scales[(r // 128) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4]
    assert torch.equal(sf_h.reshape(-1).to(torch.uint8), got_sf)
    # zero and f32-subnormal-normalised blocks quantise to exactly zero with scale byte 0
    assert int(got_sf[0]) == 0 and not q_r[0].float().any()      # +-0 only (the -a/3 lane is -0.0)
    for i, a in enumerate(amax_values):
        if 0 < a < 1e-35:        # amax / 448 is below the smallest normal f32
            assert int(got_sf[i]) in (0, 1)
        elif a >= 1e-30:
            assert int(got_sf[i]) == independent_sf(float(torch.tensor(a).to(torch.bfloat16)))


def test_swizzle_layout_and_padding():
    t, g = 130, 160 // 32
    q, scales = cm.mxfp8_quant_reference(torch.randn(t, 160).to(torch.bfloat16))
    assert scales.numel() == cm.mxfp8_scale_bytes(t, 160) == 256 * 8
    off = lambda r, gi: ((r // 128) * 2 + gi // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + gi % 4
    assert (off(0, 0), off(1, 0), off(32, 0), off(0, 1), off(0, 4), off(128, 0)) == (0, 16, 4, 1, 512, 1024)
    used = torch.zeros(scales.numel(), dtype=torch.bool)
    for r in range(t):
        for gi in range(g):
            assert not used[off(r, gi)]
            used[off(r, gi)] = True
    assert not scales[~used].any(), "padding rows / group columns are zero"
    # the device helper is the same formula
    rows, groups = torch.arange(t), torch.arange(g)
    h = cm.mxfp8_scale_offsets(rows, groups, g)
    assert h.shape == (t, g) and int(h[129, 4]) == off(129, 4) and bool(used[h.flatten()].all())


def test_scale_bytes_matches_vllm_fake_impl():
    for tokens, width in ((1, 5120), (128, 5120), (129, 1280), (6, 288), (192, 1024)):
        m_pad = -(-tokens // 128) * 128
        k_pad = -(-(width // 32) // 4) * 4
        assert cm.mxfp8_scale_bytes(tokens, width) == m_pad * k_pad   # mxfp8_e4m3_quantize_fake


# ---------------------------------------------------------------------------
# capture / replay boot gate (fake graph runner: replay = re-run the closure into the captured outputs)
# ---------------------------------------------------------------------------
class FakeRunner:
    def capture(self, fn):
        self.fn = fn
        self.out = fn()
        self.flat = cm._flat(self.out)
        return self.out

    def replay(self):
        for o, n in zip(self.flat, cm._flat(self.fn())):
            o.copy_(n)


def inputs(i):
    g = torch.Generator().manual_seed(i)
    return [torch.randn(4, 8, generator=g), torch.randn(8, generator=g).to(torch.bfloat16)]


def test_gate_passes_a_pure_kernel():
    logs = []
    ok = cm.capture_replay_gate("pure", lambda x, w: (x * w.float(), x.sum(1)), inputs,
                                runner_factory=FakeRunner, log=logs.append)
    assert ok and "pure_capture_replay PASS" in logs[0]


def test_gate_catches_state_left_behind():
    calls = {"n": 0}

    def leaky(x, w):                           # a counter that is not reset between launches
        calls["n"] += 1
        return (x * float(calls["n"] % 3),)
    logs = []
    assert not cm.capture_replay_gate("leaky", leaky, inputs, runner_factory=FakeRunner, log=logs.append)
    assert "FAIL" in logs[0] and "not bit-identical" in logs[0]


def test_gate_catches_nondeterminism():
    gen = torch.Generator().manual_seed(0)

    def noisy(x, w):                           # e.g. an atomic-order split-K: replays differ from eager
        return (x * 2 + torch.rand(x.shape, generator=gen),)
    assert not cm.capture_replay_gate("noisy", noisy, inputs, runner_factory=FakeRunner, log=lambda s: None)


def test_bits_equal_is_nan_safe_and_dtype_strict():
    a = torch.tensor([float("nan"), 1.0])
    assert cm._bits_equal(a, a.clone()) and not cm._bits_equal(a, a.to(torch.bfloat16))
    assert not cm._bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))      # bit-exact: -0 != +0
    f8 = torch.tensor([1.0, 2.0]).to(torch.float8_e4m3fn)
    assert cm._bits_equal(f8, f8.clone())
