#!/usr/bin/env python3
"""Extract a tested wheel as files for the console's existing plugin mount."""
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
