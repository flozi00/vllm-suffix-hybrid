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

if __name__ == '__main__':
    unittest.main()
