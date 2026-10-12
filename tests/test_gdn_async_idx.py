# SPDX-License-Identifier: Apache-2.0
"""CPU test for SUFFIX_ROCM_GDN_ASYNC_IDX (no vLLM, no GPU).

R_* are byte-exact excerpts of vllm/v1/attention/backends/gdn_attn.py @81198e97
(line numbers) carrying every anchor; FAKE wraps them into a runnable builder, so
the stock text (GPU[cpu_mask]) and the patched text (GPU[device row index]) run
on the same CPU tensors and must agree bit for bit. Silicon check: a mixed
(prefill + MTP verify) step's trace has no hipMemcpyWithStream under
GDNAttentionMetadataBuilder.build / update_block_table.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from suffix_hybrid import rocm_patches as rp

GATE = "SUFFIX_ROCM_GDN_ASYNC_IDX"
PATCHES = rp.PATCHES[GATE]

R_FIELD = (  # L88
    "    token_chunk_offset_ptr: torch.Tensor | None = None\n"
)
R_INIT = (  # L287
    "        spec_sequence_masks_cpu: torch.Tensor | None = None\n"
)
R_MIXED = (  # L417-443
    "                spec_state_indices_tensor = block_table_tensor[\n"
    "                    spec_sequence_masks_cpu, : self.num_spec + 1\n"
    "                ]\n"
    "                non_spec_state_indices_tensor = block_table_tensor[\n"
    "                    non_spec_sequence_masks_cpu, 0\n"
    "                ]\n"
    "\n"
    "                spec_query_start_loc = torch.zeros(\n"
    "                    num_spec_decodes + 1,\n"
    "                    dtype=torch.int32,\n"
    "                    device=query_start_loc.device,\n"
    "                )\n"
    "                torch.cumsum(\n"
    "                    query_lens[spec_sequence_masks_cpu],\n"
    "                    dim=0,\n"
    "                    out=spec_query_start_loc[1:],\n"
    "                )\n"
    "                non_spec_query_start_loc = torch.zeros(\n"
    "                    query_lens.size(0) - num_spec_decodes + 1,\n"
    "                    dtype=torch.int32,\n"
    "                    device=query_start_loc.device,\n"
    "                )\n"
    "                torch.cumsum(\n"
    "                    query_lens[non_spec_sequence_masks_cpu],\n"
    "                    dim=0,\n"
    "                    out=non_spec_query_start_loc[1:],\n"
    "                )\n"
)
R_ACCEPTED = (  # L455-458
    "            if num_prefills == 0 and num_decodes == 0:\n"
    "                num_accepted_tokens = num_accepted_tokens[:num_spec_decodes]\n"
    "            else:\n"
    "                num_accepted_tokens = num_accepted_tokens[spec_sequence_masks_cpu]\n"
)
R_INITIAL = (  # L505-509
    "        if num_prefills > 0:\n"
    "            context_lens_tensor = m.compute_num_computed_tokens()\n"
    "            has_initial_state = context_lens_tensor > 0\n"
    "            if spec_sequence_masks_cpu is not None:\n"
    "                has_initial_state = has_initial_state[~spec_sequence_masks_cpu]\n"
)
R_CTOR = (  # L621
    "            spec_sequence_masks_cpu=spec_sequence_masks_cpu,\n"
)
R_UPDATE = (  # L676-687
    "        masks = m.spec_sequence_masks_cpu\n"
    "        spec_indices = non_spec_indices = prefill_indices = None\n"
    "        if masks is None:\n"
    "            non_spec_indices = blk_table[:, 0]\n"
    "            if m.num_prefills > 0:\n"
    "                prefill_indices = non_spec_indices[m.num_decodes :]\n"
    "        elif m.num_prefills == 0:\n"
    "            # Same as build(): padded sequences trail the spec decodes.\n"
    "            spec_indices = blk_table[: m.num_spec_decodes, : self.num_spec + 1]\n"
    "        else:\n"
    "            spec_indices = blk_table[masks, : self.num_spec + 1]\n"
    "            non_spec_indices = prefill_indices = blk_table[~masks, 0]\n"
)
REAL = R_FIELD + R_INIT + R_MIXED + R_ACCEPTED + R_INITIAL + R_CTOR + R_UPDATE

# Glue around the excerpts: a mixed spec batch (num_prefills > 0, num_decodes 0).
FAKE = (
    "import torch\n"
    "from dataclasses import dataclass\n"
    "H2D = []\n"
    "def async_tensor_h2d(data, device):  # vllm.utils.torch_utils stand-in, logs calls\n"
    "    H2D.append(data)\n"
    "    return data.to(device)\n"
    "@dataclass\n"
    "class GDNAttentionMetadata:\n"
    "    num_prefills: int\n"
    "    num_decodes: int\n"
    "    num_spec_decodes: int\n"
    "    spec_sequence_masks_cpu: torch.Tensor\n"
    "    out: tuple\n"
    + R_FIELD +
    "class Builder:\n"
    "    num_spec = 2\n"
    "    def build(self, m, num_decode_draft_tokens_cpu, num_accepted_tokens, block_table_tensor):\n"
    + R_INIT +
    "        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0\n"
    "        non_spec_sequence_masks_cpu = ~spec_sequence_masks_cpu\n"
    "        num_spec_decodes = spec_sequence_masks_cpu.sum().item()\n"
    "        num_prefills, num_decodes = 1, 0\n"
    "        query_start_loc = m.query_start_loc\n"
    "        query_lens = query_start_loc[1:] - query_start_loc[:-1]\n"
    "        if True:\n"
    "            if True:\n"
    + R_MIXED + R_ACCEPTED + R_INITIAL +
    "        return GDNAttentionMetadata(\n"
    "            num_prefills=num_prefills,\n"
    "            num_decodes=num_decodes,\n"
    "            num_spec_decodes=num_spec_decodes,\n"
    + R_CTOR +
    "            out=(spec_state_indices_tensor, non_spec_state_indices_tensor,\n"
    "                 spec_query_start_loc, non_spec_query_start_loc, num_accepted_tokens,\n"
    "                 has_initial_state),\n"
    "        )\n"
    "    def update_block_table(self, m, blk_table):\n"
    + R_UPDATE +
    "        return spec_indices, non_spec_indices, prefill_indices\n"
)
# Any subscript by a mask: stock has 8 in the excerpts, patched must have none.
MASK_SUBSCRIPT = re.compile(r"\[\s*~?(?:non_)?spec_sequence_masks_cpu\b|\[\s*~?masks\b")


def patched(src):
    for p in PATCHES:
        src = rp.patch_source(p, src)
    return src


def test_anchors_bear_on_the_real_text_once(monkeypatch):
    assert len(PATCHES) == 9
    for p in PATCHES:
        assert p.target == "vllm.v1.attention.backends.gdn_attn" and not p.after
        assert REAL.count(p.old) == 1, p.label
    out = patched(REAL)
    assert len(MASK_SUBSCRIPT.findall(REAL)) == 8 and not MASK_SUBSCRIPT.findall(out)
    compile(patched(FAKE), "<gdn_attn>", "exec")
    for gate in rp.PATCHES:
        monkeypatch.delenv(gate, raising=False)
    monkeypatch.setenv(GATE, "1")
    assert rp.enabled() == {"vllm.v1.attention.backends.gdn_attn": list(PATCHES)}
    site = (Path(__file__).resolve().parents[1] / "sitecustomize.py").read_text()
    assert f'"{GATE}"' in site


def run(src, draft, qlens, accepted, ctx, block_table, blk_tables):
    ns: dict = {}
    exec(compile(src, "<gdn_attn>", "exec"), ns)
    qsl = torch.zeros(len(qlens) + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)
    m = SimpleNamespace(query_start_loc=qsl, compute_num_computed_tokens=lambda: ctx)
    b = ns["Builder"]()
    meta = b.build(m, draft, accepted, block_table)
    outs = list(meta.out)
    for blk in blk_tables:  # other GDN KV groups reuse the first group's metadata
        outs += b.update_block_table(meta, blk)
    return outs, meta, ns["H2D"]


@pytest.mark.parametrize("spec_rows, qlens", [
    ([0, 1, 2, 3, 4, 5, 6], [3, 3, 3, 3, 3, 3, 3, 100, 0, 0]),  # verify, prefill, padding
    ([1, 3, 4, 7], [1, 3, 17, 3, 2, 1, 40, 3]),  # interleaved, 1-token decodes as prefill
    ([2], [5, 9, 3]),
])
def test_patched_gathers_equal_bool_mask_gathers(spec_rows, qlens):
    n = len(qlens)
    g = torch.Generator().manual_seed(n)
    draft = torch.full((n,), -1, dtype=torch.int32)
    draft[spec_rows] = 2
    accepted = torch.randint(1, 4, (n,), dtype=torch.int32, generator=g)
    ctx = torch.arange(n, dtype=torch.int32) % 2 * 7  # every other row has state
    block_table = torch.randint(0, 1 << 20, (n, 4), dtype=torch.int32, generator=g)
    blk_tables = [torch.randint(0, 1 << 20, (n, 6), dtype=torch.int32, generator=g)[:, ::2],
                  torch.randint(0, 1 << 20, (n, 3), dtype=torch.int32, generator=g)]
    args = (draft, qlens, accepted, ctx, block_table, blk_tables)

    stock, _, stock_h2d = run(FAKE, *args)
    new, meta, h2d = run(patched(FAKE), *args)
    assert len(stock) == len(new) == 12
    for a, b in zip(stock, new):
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
        assert torch.equal(a, b)
    assert stock_h2d == [] and len(h2d) == 2  # once per build, none per extra group
    assert meta.spec_req_idx.dtype == torch.int64
    assert meta.spec_req_idx.tolist() == spec_rows
    assert meta.non_spec_req_idx.tolist() == sorted(set(range(n)) - set(spec_rows))
