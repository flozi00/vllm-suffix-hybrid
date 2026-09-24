# SPDX-License-Identifier: Apache-2.0
"""CPU check of the on-silicon oracle's reference decoder: if dequant_side is
wrong, the GPU oracle would compare the kernel against garbage. Bytes are laid
out here with the patch's independent layout helpers (swizzle_scale_offset
mirrors nvfp4_kv_cache_kernels.cu), then decoded by the oracle."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvfp4_kv_patch import SF_VEC_SIZE, swizzle_scale_offset  # noqa: E402
from nvfp4_kv_patch.oracle import _E2M1, dequant_side  # noqa: E402


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("swizzled", [False, True])
def test_dequant_side_matches_store_layout(head_dim, swizzled):
    g = torch.Generator().manual_seed(head_dim + swizzled)
    pages, heads, n = 2, 3, 8
    s_dim = head_dim // SF_VEC_SIZE
    codes = torch.randint(0, 16, (pages, heads, n, head_dim), generator=g)
    # Powers of two are exact in e4m3.
    sf = 2.0 ** torch.randint(-6, 6, (pages, heads, n, s_dim), generator=g)

    data = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    phys = sf.clone()
    if swizzled:
        flat = phys.view(pages, heads, n * s_dim)
        for t in range(n):
            for s in range(s_dim):
                flat[:, :, swizzle_scale_offset(t, s, s_dim)] = sf[:, :, t, s]
    phys = phys.to(torch.float8_e4m3fn)

    got = dequant_side(data, phys, swizzled, 0.5)
    want = (torch.tensor(_E2M1)[codes]
            * sf.repeat_interleave(SF_VEC_SIZE, dim=-1) * 0.5)
    assert torch.equal(got, want)
