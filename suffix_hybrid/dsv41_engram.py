# SPDX-License-Identifier: Apache-2.0
"""MXFP4 Engram tables for DeepSeek-V4.1 (``SUFFIX_DSV41_ENGRAM_MXFP4=1``).

Checkpoints in NVIDIA's DeepSeek-V4.1-Flash-NVFP4 layout (also
LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4) store the Engram tables as packed
MXFP4 (``engram_embed_quant: "mxfp4"``). Stock vLLM allocates them as FP8 and
cannot load them until vllm-project/vllm#61530 lands. This ships that PR's
engram.py (head 1800929afd, 3-way ported onto nightly e0d00e4b8; peer-HBM
tables refuse MXFP4) and execs it over
the stock module right after its first import.

Fail closed: the stock file must hash to BASE_SHA256 (the file the PR was
applied to). A vLLM that already reads ``engram_embed_quant`` is upstream
support: inert with one log line. Anything else refuses to start.
"""

import hashlib
import os
import sys
from pathlib import Path

GATE_ENV = "SUFFIX_DSV41_ENGRAM_MXFP4"
TARGET = "vllm.models.deepseek_v41.common.engram"
BASE_SHA256 = "e10681d8cb573cd0f192384dbe47786eb5d1e3809996f827eed48b9aa690365c"
PATCHED = Path(__file__).with_name("dsv41_engram_pr61530.txt")
TAG = "[suffix dsv41-engram-mxfp4]"


def apply(module) -> bool:
    src_path = Path(module.__file__)
    src = src_path.read_bytes()
    if b"engram_embed_quant" in src:
        print(f"{TAG} inert: {src_path} already reads engram_embed_quant (upstream #61530).",
              file=sys.stderr, flush=True)
        return False
    digest = hashlib.sha256(src).hexdigest()
    if digest != BASE_SHA256:
        raise RuntimeError(f"{src_path} sha256 {digest} != pinned base {BASE_SHA256}: "
                           "re-port vllm#61530 onto this vLLM")
    new_src = PATCHED.read_text()
    fname = f"{src_path}.dsv41-engram-mxfp4.py"
    import linecache
    linecache.cache[fname] = (len(new_src), None, new_src.splitlines(keepends=True), fname)
    exec(compile(new_src, fname, "exec"), module.__dict__)
    print(f"{TAG} ACTIVE: {TARGET} replaced by vllm#61530 (MXFP4 tables).",
          file=sys.stderr, flush=True)
    return True


def _callback(module) -> None:
    try:
        apply(module)
    except Exception as exc:
        raise SystemExit(f"{TAG} enabled but FAILED: {exc}") from exc


def install_post_import_hook() -> bool:
    if os.environ.get(GATE_ENV, "").strip() != "1":
        return False
    if TARGET in sys.modules:
        raise SystemExit(f"{TAG} {TARGET} imported before the hook armed")
    from nvfp4_kv_patch import _PostImportFinder
    sys.meta_path.insert(0, _PostImportFinder(TARGET, _callback))
    return True
