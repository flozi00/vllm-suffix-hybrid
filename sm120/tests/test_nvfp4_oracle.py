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


def test_sitecustomize_warmup_oracle(tmp_path):
    # Off-SM120 the oracle exits 2 (NOT RUN): logged-only unless the pod
    # serves NVFP4 KV, then fatal. Runs once: the child must not recurse.
    import os
    import shutil
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    bundle = tmp_path / "plugins"
    shutil.copytree(repo / "sm120" / "nvfp4_kv_patch",
                    bundle / "nvfp4_kv_patch")
    shutil.copy(repo / "sitecustomize.py", bundle / "sitecustomize.py")
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUFFIX")}
    env.update(PYTHONPATH=str(bundle), SUFFIX_SM120_NVP4KV_ORACLE="1")

    def run():
        return subprocess.run([sys.executable, "-c", "print('SERVING')"],
                              capture_output=True, text=True, env=env,
                              timeout=120)

    soft = run()
    assert soft.returncode == 0 and "SERVING" in soft.stdout, soft.stderr
    assert soft.stderr.count("oracle: running") == 1       # no recursion
    assert "oracle: exit 2 (NOT RUN)" in soft.stderr
    env["SUFFIX_SM120_NVP4KV"] = "1"
    hard = run()
    assert hard.returncode != 0 and "SERVING" not in hard.stdout
    assert "refusing to serve NVFP4 KV" in hard.stderr
