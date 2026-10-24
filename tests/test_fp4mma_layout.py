# SPDX-License-Identifier: Apache-2.0
"""K1 for the native-FP4-MMA family: the scale-layout contract, byte for byte.

The swizzled (128x4, FlashInfer / cuBLAS SWIZZLE_32_4_4) scale buffer stores the four scales of one
(row, k64 step) as ONE aligned little-endian 32-bit word; the bodies gather those words and unpack them in
registers. Every index formula the bodies use is pinned here against an independently built buffer, and the
in-kernel unpack (shift / mask / hl.join order) against torch.
"""
import sys

import numpy as np
import pytest
import torch

from suffix_hybrid.helion_kernels import nvfp4_gemm_fp4mma as fm
from suffix_hybrid.kernels import nvfp4_gemm as ng

from test_fp4mma_ref import adversarial_scales  # noqa: E402


def swizzled(sf: np.ndarray) -> np.ndarray:
    """An INDEPENDENT builder of the vLLM swizzle (written from the layout text, not from
    kernels/nvfp4_gemm.swizzle_sf): [R, C] -> flat, rows padded to 128, cols to 4."""
    r, c = sf.shape
    rp, cp = -(-r // 128) * 128, -(-c // 4) * 4
    out = np.zeros(rp * cp, np.uint8)
    for row in range(r):
        for kb in range(c):
            block = row // 128
            q = kb // 4
            within = (row % 32) * 16 + ((row // 32) % 4) * 4 + kb % 4
            out[(block * (cp // 4) + q) * 512 + within] = sf[row, kb]
    return out


SHAPES = [(1, 4), (5, 8), (128, 4), (129, 12), (200, 16), (257, 20), (300, 64)]


@pytest.mark.parametrize("rows,nkb", SHAPES)
def test_swizzle_matches_the_spec_builder_and_the_byte_index(rows, nkb):
    sf = adversarial_scales(rows, nkb, rows + nkb).numpy()
    flat = swizzled(sf)
    np.testing.assert_array_equal(flat, ng.swizzle_sf(sf))  # independent builder == the lanes' spec
    r, kb = np.meshgrid(np.arange(rows), np.arange(nkb), indexing="ij")
    np.testing.assert_array_equal(flat[fm.swz_byte_index(r, kb, nkb)], sf)
    np.testing.assert_array_equal(fm.swz_byte_index(r, kb, nkb), ng.sf_offset(r, kb, -(-nkb // 4) * 4))


@pytest.mark.parametrize("rows,nkb", SHAPES)
def test_word_index_addresses_the_four_scales_of_a_k64_step(rows, nkb):
    """words[swz_word_index(row, q)] holds blocks 4q..4q+3 of `row`, byte j = block 4q + j, for EVERY
    (row, quad) incl. the padded rows of the last 128-row atom; blocks past nkb read the zero pad."""
    sf = adversarial_scales(rows, nkb, 7 * rows + nkb).numpy()
    flat = swizzled(sf)
    words = flat.view("<u4")  # little-endian words of the byte buffer
    nq = -(-nkb // 4)
    padded = np.zeros((rows, nq * 4), np.uint8)
    padded[:, :nkb] = sf
    r, q = np.meshgrid(np.arange(rows), np.arange(nq), indexing="ij")
    w = words[fm.swz_word_index(r, q, nkb)]
    for j in range(4):
        np.testing.assert_array_equal(((w >> (8 * j)) & 255).astype(np.uint8), padded[:, 4 * q[0] + j])
    # the byte offset of the word is the byte offset of block 4q, 4-aligned
    np.testing.assert_array_equal(fm.swz_word_index(r, q, nkb) * 4, fm.swz_byte_index(r, 4 * q, nkb))


def test_word_index_at_the_v41_shapes():
    """V4.1 EP8: w13 rows 2I = 4608 (36 whole atoms) x H/16 = 320 blocks; w2 5120 x 144; lm_head 16160 x 320
    (16160 = 126.25 atoms: the last atom is partly pad). Sampled rows incl. the first / last of each atom."""
    for rows, nkb in ((4608, 320), (5120, 144), (16160, 320)):
        rng = np.random.default_rng(rows)
        sf = rng.integers(0, 0x7F, size=(rows, nkb)).astype(np.uint8)
        words = swizzled(sf).view("<u4")
        pick = np.unique(np.concatenate([rng.integers(0, rows, 300), np.arange(0, rows, 128),
                                         np.arange(127, rows, 128), [rows - 1]]))
        q = rng.integers(0, nkb // 4, size=pick.shape)
        w = words[fm.swz_word_index(pick, q, nkb)]
        for j in range(4):
            np.testing.assert_array_equal(((w >> (8 * j)) & 255).astype(np.uint8), sf[pick, 4 * q + j])


def test_per_expert_word_stride():
    """The MoE bodies address expert e's scale matrix at e * round128(rows) * (K/16) / 4 words."""
    e_count, rows, nkb = 3, 200, 16
    sfs = [adversarial_scales(rows, nkb, 40 + e).numpy() for e in range(e_count)]
    flat = np.concatenate([swizzled(s) for s in sfs]).view("<u4")
    ew = (rows + 127) // 128 * 128 * (nkb // 4)
    for e in range(e_count):
        r, q = np.meshgrid(np.arange(rows), np.arange(nkb // 4), indexing="ij")
        w = flat[e * ew + fm.swz_word_index(r, q, nkb)]
        for j in range(4):
            np.testing.assert_array_equal(((w >> (8 * j)) & 255).astype(np.uint8), sfs[e][:, 4 * q[0] + j])


@pytest.mark.parametrize("rows,nkb", [(5, 8), (9, 16), (64, 20)])
def test_linear_word_index(rows, nkb):
    sf = adversarial_scales(rows, nkb, 3).numpy()
    words = sf.reshape(-1).view("<u4")
    r, q = np.meshgrid(np.arange(rows), np.arange(nkb // 4), indexing="ij")
    w = words[fm.lin_word_index(r, q, nkb)]
    for j in range(4):
        np.testing.assert_array_equal(((w >> (8 * j)) & 255).astype(np.uint8), sf[:, 4 * q[0] + j])


def test_little_endian_is_asserted():
    assert sys.byteorder == "little"  # as_words / the unpack assume byte j of a word = block 4q + j


def test_as_words_is_the_same_bytes_and_rejects_the_unsafe():
    sf = torch.from_numpy(swizzled(adversarial_scales(130, 8, 1).numpy()))
    w = fm.as_words(sf)
    assert w.dtype == torch.int32 and w.numel() == sf.numel() // 4
    assert w.data_ptr() == sf.data_ptr()  # a view, not a copy
    assert torch.equal(fm.as_words(sf.view(torch.float8_e4m3fn)), w)
    assert fm.as_words(w) is not None and torch.equal(fm.as_words(w), w)
    back = fm.words_to_bytes(w).reshape(-1)
    assert torch.equal(back, sf)
    with pytest.raises(RuntimeError, match="contiguous"):
        fm.as_words(sf.reshape(-1, 4)[:, :2])
    with pytest.raises(RuntimeError, match="numel"):
        fm.as_words(sf[:-1])
    with pytest.raises(RuntimeError, match="dtype"):
        fm.as_words(sf.float())


def test_unpack_order_hl_join_equals_words_to_bytes():
    """The body's `hl.join(hl.join(w & 255, (w >> 16) & 255), hl.join((w >> 8) & 255, (w >> 24) & 255))`
    then reshape gives bytes in k order (join appends a minor dim: [..., 2, 2] row-major = b0 b1 b2 b3)."""
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (7, 5), dtype=torch.int32)
    join = lambda a, b: torch.stack([a, b], -1)  # the eager semantic of hl.join
    got = join(join(w & 255, (w >> 16) & 255), join((w >> 8) & 255, (w >> 24) & 255)).reshape(7, -1)
    want = fm.words_to_bytes(w).reshape(7, -1).to(torch.int32)
    assert torch.equal(got, want)
    # arithmetic >> on a negative word (top byte >= 0x80, never a valid scale): the mask still isolates the byte
    neg = torch.tensor([-1, -2 ** 31], dtype=torch.int32)
    assert ((neg >> 24) & 255).tolist() == [255, 128]


def test_linearize_roundtrip_and_padding_rows_dropped():
    sf = adversarial_scales(130, 12, 5)
    lin = fm.linearize(torch.from_numpy(swizzled(sf.numpy())), 130, 12)
    assert torch.equal(lin, sf)


def test_validate_scales_fails_closed():
    ok = adversarial_scales(8, 8)
    fm.validate_scales(ok)
    for bad in (0x7F, 0xFF, 0x80, 0xBF):
        s = ok.clone()
        s[3, 3] = bad
        with pytest.raises(RuntimeError, match="0x7F"):
            fm.validate_scales(s, "w13")


try:
    import helion
    import helion.language as hl
except ImportError:  # no Helion: the eager gather test skips
    helion = hl = None


def gather_body(wsf: torch.Tensor, out: torch.Tensor):
    """The scale gather + unpack of nvfp4_gemm_q_fp4mma, alone: out[row, 4q + j] = byte j of word q."""
    n = out.size(0)
    cp4 = hl.specialize(out.size(1)) // 4
    rw = cp4 * 128
    for tile_n in hl.tile(n, block_size=n):
        rn = torch.clamp(tile_n.index, max=n - 1)[:, None]
        rb = (rn >> 7) * rw + (rn & 31) * 4 + ((rn >> 5) & 3)
        for tile_q in hl.tile(cp4, block_size=cp4):
            q = (torch.clamp(tile_q.index, max=cp4 - 1) * 128)[None, :]
            wb = wsf[rb + q]
            sb = hl.join(hl.join(wb & 255, (wb >> 16) & 255), hl.join((wb >> 8) & 255, (wb >> 24) & 255))
            out[tile_n, :] = sb.reshape(sb.size(0), -1).to(torch.uint8)
    return out


def test_helion_gather_with_the_bodies_index_expressions_reads_the_right_words():
    """The body's own index maths through Helion's eager mode: gather the scale tile of every row and
    k64 step with the exact expressions of nvfp4_gemm_q_fp4mma and compare with the logical matrix."""
    pytest.importorskip("helion")
    rows, nkb = 150, 24
    sf = adversarial_scales(rows, nkb, 11)
    words = fm.as_words(torch.from_numpy(swizzled(sf.numpy())))
    out = torch.zeros(rows, nkb, dtype=torch.uint8)
    got = helion.kernel(gather_body, config=helion.Config(), static_shapes=False)(words, out)
    assert torch.equal(got, sf)
