"""Prepare bounded Windows download artifacts without combining large binaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PART_BYTES = 300 * 1024 * 1024
INSTALLER_MAX_BYTES = 511 * 1024 * 1024  # Leave room for the artifact ZIP wrapper.


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def prepare(directory: Path, version='3.0.0', *, part_bytes=PART_BYTES):
    directory = Path(directory)
    if not 1 <= part_bytes <= PART_BYTES:
        raise ValueError('Part size must be between1 byte and300 MiB.')
    installer = directory / f'SARU-Picker-Setup-{version}-win64.exe'
    portable = directory / f'SARU-Picker-GUI-{version}-win64.zip'
    if installer.exists() and installer.stat().st_size >= INSTALLER_MAX_BYTES:
        raise ValueError('The installer alone exceeds the connector download budget; do not combine or publish it as a larger artifact.')
    if not portable.is_file():
        raise ValueError('The verified portable ZIP is missing.')
    part_count = (portable.stat().st_size + part_bytes - 1) // part_bytes
    if not 1 <= part_count <= 4:
        raise ValueError('The portable ZIP must fit in one to four bounded download parts.')
    destination = directory / 'portable-parts'
    destination.mkdir(exist_ok=True)
    # A repeat preparation only replaces files with this helper's fixed names.
    for index in range(1, 5):
        path = destination / f'portable.part{index:02d}'
        if path.exists():
            path.unlink()
    parts = []
    combined = hashlib.sha256()
    with portable.open('rb') as source:
        for index in range(1, part_count + 1):
            path = destination / f'portable.part{index:02d}'
            checksum = hashlib.sha256()
            written = 0
            with path.open('wb') as target:
                while written < part_bytes:
                    chunk = source.read(min(8 * 1024 * 1024, part_bytes - written))
                    if not chunk:
                        break
                    target.write(chunk)
                    checksum.update(chunk)
                    combined.update(chunk)
                    written += len(chunk)
            parts.append({'filename': path.name, 'bytes': written, 'sha256': checksum.hexdigest()})
    manifest = {
        'format': 'saru-windows-downloads-v1', 'version': version,
        'installer': ({'filename': installer.name, 'bytes': installer.stat().st_size,
                       'sha256': digest(installer)} if installer.exists() else None),
        'portable': {'filename': portable.name, 'bytes': portable.stat().st_size,
                     'sha256': combined.hexdigest(), 'parts': parts},
        'note': 'Installer and verification reports are separate downloads. Portable ZIP parts are optional; concatenate in order and verify the original hash before extracting.',
    }
    (directory / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--version', default='3.0.0')
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.version), indent=2))
