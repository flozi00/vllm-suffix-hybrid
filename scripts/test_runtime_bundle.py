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
            shipped = (root / 'runtime/deep_gemm/__init__.py').read_text()
            self.assertIn('__suffix_shim__', shipped)

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
