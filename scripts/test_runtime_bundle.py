import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

class BundleTest(unittest.TestCase):
    def test_bundle_contains_native_module_and_verifiable_manifest(self):
        spec = importlib.util.spec_from_file_location('runtime_bundle', Path(__file__).with_name('runtime_bundle.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'plugin.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('suffix_hybrid/__init__.py', '')
                archive.writestr('suffix_hybrid/_native.abi3.so', b'test-fixture-not-a-real-library')
                archive.writestr('plugin.dist-info/METADATA', 'metadata')
            module.bundle(wheel, root / 'runtime', 'a' * 40)
            manifest = json.loads((root / 'runtime/BUILD.json').read_text())
            self.assertEqual(manifest['source_revision'], 'a' * 40)
            self.assertIn('suffix_hybrid/_native.abi3.so', manifest['sha256'])
            self.assertFalse((root / 'runtime/plugin.dist-info').exists())

    def test_bundle_ships_sm120_shim_at_top_level(self):
        # The shim must land at <bundle>/deep_gemm/ (top level) to shadow
        # site-packages via PYTHONPATH=/plugins, and be hashed in BUILD.json.
        spec = importlib.util.spec_from_file_location('runtime_bundle', Path(__file__).with_name('runtime_bundle.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'plugin.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('suffix_hybrid/__init__.py', '')
                archive.writestr('suffix_hybrid/_native.abi3.so', b'test-fixture-not-a-real-library')
            module.bundle(wheel, root / 'runtime', 'b' * 40)
            manifest = json.loads((root / 'runtime/BUILD.json').read_text())
            self.assertIn('deep_gemm/__init__.py', manifest['sha256'])
            self.assertIn('deep_gemm/sm120_fallback.py', manifest['sha256'])
            self.assertIn('hisparse_mtp_patch/__init__.py', manifest['sha256'])
            self.assertIn('hisparse_mtp_patch/oracle.py', manifest['sha256'])
            self.assertIn('nvfp4_ds_mla_patch/__init__.py', manifest['sha256'])
            self.assertIn('nvfp4_ds_mla_patch/oracle.py', manifest['sha256'])
            shipped = (root / 'runtime/deep_gemm/__init__.py').read_text()
            self.assertIn('__suffix_shim__', shipped)

    def test_bundle_ships_sm120_ficache_flat(self):
        # ficache ships FLAT at <bundle>/ficache.py (sitecustomize loads it by
        # file location), hashed in BUILD.json; seed payloads are optional.
        spec = importlib.util.spec_from_file_location('runtime_bundle', Path(__file__).with_name('runtime_bundle.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'plugin.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('suffix_hybrid/__init__.py', '')
                archive.writestr('suffix_hybrid/_native.abi3.so', b'test-fixture-not-a-real-library')
            module.bundle(wheel, root / 'runtime', 'd' * 40)
            manifest = json.loads((root / 'runtime/BUILD.json').read_text())
            self.assertIn('ficache.py', manifest['sha256'])
            shipped = (root / 'runtime/ficache.py').read_text()
            self.assertIn('SUFFIX_FICACHE_DUMP', shipped)
            self.assertIn('resolve_flashinfer_autotune_file', shipped)

    def test_ficache_dump_roundtrip_and_seed(self):
        import base64
        import gzip
        import sys
        fic = Path(__file__).resolve().parents[1] / 'sm120' / 'ficache' / '__init__.py'
        spec = importlib.util.spec_from_file_location('suffix_ficache', fic)
        ficache = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ficache)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = json.dumps({'tactics': {'trtllm::fused_moe::gemm1': 21}}).encode()
            cfg = root / 'deadbeef' / 'autotune_configs.json'
            cfg.parent.mkdir()
            cfg.write_bytes(payload)
            # dump: encode + decode roundtrip
            line = ficache.encode_dump(cfg)
            marker, hash_dir, b64 = line.split(' ')
            self.assertEqual(marker, 'SUFFIX_FICACHE_DUMP')
            self.assertEqual(hash_dir, 'deadbeef')
            self.assertEqual(gzip.decompress(base64.b64decode(b64)), payload)
            # seed: matching hash name seeds; mismatch no-ops; corrupt no-ops
            target = root / 'live' / 'deadbeef' / 'autotune_configs.json'
            target.parent.mkdir(parents=True)
            seeds = root / 'seeds'
            seeds.mkdir()
            ficache._SEEDS_DIR = seeds
            (seeds / 'deadbeef.json.gz').write_bytes(gzip.compress(payload))
            ficache._seed_if_missing(target)
            self.assertEqual(target.read_bytes(), payload)
            target.unlink()
            ficache._SEEDS_DIR = seeds / 'nonexistent'
            ficache._seed_if_missing(target)  # missing seeds dir -> no-op
            self.assertFalse(target.exists())
            ficache._SEEDS_DIR = seeds
            (seeds / 'cafebabe.json.gz').write_bytes(gzip.compress(b'{not json'))
            ficache._seed_if_missing(root / 'live' / 'cafebabe' / 'autotune_configs.json')
            self.assertFalse((root / 'live' / 'cafebabe' / 'autotune_configs.json').exists())

    def test_ficache_seed_hook_wraps_module(self):
        # Simulate the import hook: a fake vllm cache module gets its
        # resolve_flashinfer_autotune_file wrapped so a matching seed lands.
        import types
        fic = Path(__file__).resolve().parents[1] / 'sm120' / 'ficache' / '__init__.py'
        spec = importlib.util.spec_from_file_location('suffix_ficache2', fic)
        ficache = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ficache)
        calls = []

        def fake_resolve(runner):
            calls.append(runner)
            from pathlib import Path as P
            return P('/tmp/does-not-exist-ficache-test/hshsh/autotune_configs.json')

        mod = types.ModuleType('fake')
        mod.resolve_flashinfer_autotune_file = fake_resolve
        ficache._patch_resolve(mod)
        self.assertIsNot(mod.resolve_flashinfer_autotune_file, fake_resolve)
        # wrapping is idempotent-safe to call (seed dir absent -> no-op, no raise)
        ficache._SEEDS_DIR = __import__('pathlib').Path('/nonexistent-seed-dir')
        out = mod.resolve_flashinfer_autotune_file('runner')
        self.assertEqual(calls, ['runner'])
        self.assertTrue(str(out).endswith('autotune_configs.json'))

    def test_bundle_rejects_missing_native(self):
        spec = importlib.util.spec_from_file_location('runtime_bundle', Path(__file__).with_name('runtime_bundle.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / 'plugin.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('suffix_hybrid/__init__.py', '')
            with self.assertRaises(ValueError):
                module.bundle(wheel, root / 'runtime', 'c' * 40)

if __name__ == '__main__':
    unittest.main()
