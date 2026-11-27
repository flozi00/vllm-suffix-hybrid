# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 on SM120: 64-token kernel blocks (``SUFFIX_DSV41_SM120_BLOCK64=1``).

vllm-project/vllm#60762 (open; validated on 8x RTX PRO 6000 TP8 with FlashInfer
>= 0.7.1rc5). Stock nightly 39983a5e7 declares 128-token kernel blocks for the
V4.1 indexer and FlashInfer sparse-MLA backends on SM120; ratio-1 indexer pages
then hold 128 rows and DeepGEMM's SM120 paged-MQA kernel asserts
``block_kv == 32 or block_kv == 64``. Declare [64] on SM12x, as the PR does;
serve with ``--block-size 64``. Fail closed if a target class is missing.
"""

import os
import sys

GATE_ENV = "SUFFIX_DSV41_SM120_BLOCK64"
TARGETS = {
    "vllm.v1.attention.backends.mla.indexer": "DeepseekV41IndexerBackend",
    "vllm.models.deepseek_v41.nvidia.flashinfer_sparse": "DeepseekV4FlashInferMLASparseBackend",
}
TAG = "[suffix dsv41-sm120-block64]"


def apply(module) -> None:
    cls = getattr(module, TARGETS[module.__name__])
    stock = cls.__dict__["get_supported_kernel_block_sizes"].__func__

    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        from vllm.platforms import current_platform
        if current_platform.is_device_capability_family(120):
            return [64]
        return stock(kv_cache_spec)

    cls.get_supported_kernel_block_sizes = staticmethod(get_supported_kernel_block_sizes)
    print(f"{TAG} ACTIVE: {cls.__name__} declares 64-token kernel blocks on SM12x (vllm#60762).",
          file=sys.stderr, flush=True)


def _callback(module) -> None:
    try:
        apply(module)
    except Exception as exc:
        raise SystemExit(f"{TAG} enabled but FAILED on {module.__name__}: {exc!r}") from exc


def install_post_import_hook() -> bool:
    if os.environ.get(GATE_ENV, "").strip() != "1":
        return False
    early = [t for t in TARGETS if t in sys.modules]
    if early:
        raise SystemExit(f"{TAG} {early} imported before the hook armed")
    from nvfp4_kv_patch import _PostImportFinder
    for target in TARGETS:
        sys.meta_path.insert(0, _PostImportFinder(target, _callback))
    return True
