#!/usr/bin/env python3
"""Extract a tested wheel as files for the console's existing plugin mount.

Also ships the sm120 deep_gemm shim: sm120/deep_gemm_shim/ is copied to
<output>/deep_gemm/ so /plugins/deep_gemm shadows any site-packages copy
(PYTHONPATH=/plugins precedes site-packages, and vLLM's _import_deep_gemm
prefers an external deep_gemm over its vendored one). The shim self-gates on
SUFFIX_SM120 / vendor health, so shipping it unconditionally is safe.

And ships the sm120 NVFP4-KV patch: sm120/nvfp4_kv_patch/ is copied to
<output>/nvfp4_kv_patch/ so `import nvfp4_kv_patch` resolves from /plugins at
sitecustomize time (only ever when SUFFIX_SM120_NVP4KV=1). It rewrites the
vLLM FlashInfer backend IN MEMORY (no site-packages writes); the gate keeps it
inert everywhere else, so shipping it unconditionally is safe.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def bundle(wheel, output, revision):
    output = Path(output)
    hashes = {}
    with zipfile.ZipFile(wheel) as archive:
        members = [n for n in archive.namelist() if n.startswith('suffix_hybrid/') and not n.endswith('/')]
        if not any(n.endswith('.so') for n in members):
            raise ValueError('runtime bundle requires a compiled native extension')
        for name in members:
            if '..' in PurePosixPath(name).parts:
                raise ValueError('unsafe wheel path')
            data = archive.read(name)
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
    bootstrap = Path(__file__).resolve().parents[1] / 'sitecustomize.py'
    data = bootstrap.read_bytes()
    (output / 'sitecustomize.py').write_bytes(data)
    hashes['sitecustomize.py'] = hashlib.sha256(data).hexdigest()
    # sm120 bundle members: repo dir -> top-level bundle package (directory
    # shadowing / /plugins import is the whole point).
    #   deep_gemm_shim  -> <output>/deep_gemm/       (SUFFIX_SM120 gate)
    #   nvfp4_kv_patch  -> <output>/nvfp4_kv_patch/  (SUFFIX_SM120_NVP4KV gate)
    # Both self-gate at import time, so shipping them unconditionally is safe.
    sm120_root = Path(__file__).resolve().parents[1] / 'sm120'
    for shim_name, member in (('deep_gemm_shim', 'deep_gemm'),
                              ('nvfp4_kv_patch', 'nvfp4_kv_patch')):
        shim_root = sm120_root / shim_name
        for src in sorted(shim_root.rglob('*')):
            if src.is_dir() or '__pycache__' in src.parts:
                continue
            rel = PurePosixPath(member, *src.relative_to(shim_root).parts)
            if '..' in rel.parts:
                raise ValueError('unsafe shim path')
            data = src.read_bytes()
            dest = output / PurePosixPath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    if not (output / 'deep_gemm' / '__init__.py').exists():
        raise ValueError('runtime bundle requires the sm120 deep_gemm shim')
    if not (output / 'nvfp4_kv_patch' / '__init__.py').exists():
        raise ValueError('runtime bundle requires the sm120 nvfp4_kv patch')
    # sm120 ficache: FlashInfer autotune cache seed/harvest, shipped FLAT as
    # <output>/ficache.py (sitecustomize loads it by file location), plus the
    # optional seed payloads sm120/ficache/seeds/* -> <output>/ficache/seeds/*.
    ficache_root = Path(__file__).resolve().parents[1] / 'sm120' / 'ficache'
    ficache_mod = ficache_root / '__init__.py'
    if not ficache_mod.is_file():
        raise ValueError('runtime bundle requires the sm120 ficache module')
    data = ficache_mod.read_bytes()
    (output / 'ficache.py').write_bytes(data)
    hashes['ficache.py'] = hashlib.sha256(data).hexdigest()
    # warm-start endpoint plugin (vLLM 0.30.0 native vllm.endpoint_plugins
    # seam): the module + dist-info must ship TOP-LEVEL at <bundle>/ so that
    # with the bundle mounted at /plugins (on PYTHONPATH) importlib.metadata
    # discovers suffix_hybrid_warmstart_ep-1.0.dist-info/ and its
    # [vllm.endpoint_plugins] entry point suffix_hybrid_warmstart =
    # suffix_hybrid_warmstart_ep:register. vLLM's split-allowlist loader runs
    # ONLY when env VLLM_PLUGINS names the plugin (loader not called at all
    # when unset), and the plugin self-gates on SUFFIX_HYBRID_WARMSTART=1 —
    # so shipping it unconditionally is safe (bare file on disk = zero code
    # paths executed).
    repo = Path(__file__).resolve().parents[1]
    warmstart_ep = repo / 'suffix_hybrid_warmstart_ep.py'
    if not warmstart_ep.is_file():
        raise ValueError('runtime bundle requires suffix_hybrid_warmstart_ep')
    data = warmstart_ep.read_bytes()
    (output / 'suffix_hybrid_warmstart_ep.py').write_bytes(data)
    hashes['suffix_hybrid_warmstart_ep.py'] = hashlib.sha256(data).hexdigest()
    distinfo = repo / 'suffix_hybrid_warmstart_ep-1.0.dist-info'
    for src in sorted(distinfo.glob('*')):
        if not src.is_file():
            continue
        rel = PurePosixPath('suffix_hybrid_warmstart_ep-1.0.dist-info',
                            src.name)
        data = src.read_bytes()
        dest = output / PurePosixPath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    if not (output / 'suffix_hybrid_warmstart_ep-1.0.dist-info'
            / 'entry_points.txt').exists():
        raise ValueError('runtime bundle requires warm-start dist-info '
                        'entry_points.txt')
    # K-GDN1 (qwen3.8-27b fused GDN decode kernel): vllm.general_plugins
    # entry point suffix_qwen_gdn = suffix_hybrid.kernels.qwen_gdn:register,
    # discovered from this top-level dist-info. The plugin self-gates on
    # SUFFIX_QWEN_GDN=1 (register() returns before importing anything), so
    # shipping it unconditionally is safe. NOTE: if a pod sets VLLM_PLUGINS
    # (allowlist), it must name suffix_qwen_gdn or the plugin never loads.
    gdn_distinfo = repo / 'suffix_qwen_gdn_ep-1.0.dist-info'
    for src in sorted(gdn_distinfo.glob('*')):
        if not src.is_file():
            continue
        rel = PurePosixPath('suffix_qwen_gdn_ep-1.0.dist-info', src.name)
        data = src.read_bytes()
        dest = output / PurePosixPath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    if not (output / 'suffix_qwen_gdn_ep-1.0.dist-info'
            / 'entry_points.txt').exists():
        raise ValueError('runtime bundle requires the suffix_qwen_gdn '
                        'dist-info entry_points.txt')
    seeds_root = ficache_root / 'seeds'
    if seeds_root.is_dir():
        for src in sorted(seeds_root.rglob('*')):
            if src.is_dir():
                continue
            rel = PurePosixPath('ficache/seeds', src.name)
            data = src.read_bytes()
            dest = output / PurePosixPath(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    (output / 'BUILD.json').write_text(json.dumps({
        'source_revision': revision,
        'wheel': Path(wheel).name,
        'sha256': hashes,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    bundle(args.wheel, args.output, args.revision)
