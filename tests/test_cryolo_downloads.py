"""Transfer bounds and exact reconstruction for Windows release artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'packaging' / 'cryolo' / 'prepare_downloads.py'
spec = importlib.util.spec_from_file_location('cryolo_downloads', MODULE)
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


def test_parts_reassemble_exact_portable_bytes_and_installer_stays_separate(tmp_path):
    installer = tmp_path / 'SARU-Picker-Setup-3.0.0-win64.exe'
    installer.write_bytes(b'installer fixture')
    portable = tmp_path / 'SARU-Picker-GUI-3.0.0-win64.zip'
    data = bytes(range(256)) * 3 + b'last short part'
    portable.write_bytes(data)
    result = downloads.prepare(tmp_path, part_bytes=300)
    parts = result['portable']['parts']
    assert [item['bytes'] for item in parts] == [300, 300, len(data) - 600]
    assert b''.join((tmp_path / 'portable-parts' / item['filename']).read_bytes() for item in parts) == data
    assert result['portable']['sha256'] == hashlib.sha256(data).hexdigest()
    assert result['installer']['sha256'] == hashlib.sha256(installer.read_bytes()).hexdigest()
    assert json.loads((tmp_path / 'download-manifest.json').read_text()) == result
    for item in parts:
        assert hashlib.sha256((tmp_path / 'portable-parts' / item['filename']).read_bytes()).hexdigest() == item['sha256']


def test_oversized_installer_is_rejected_before_hashing_or_upload(tmp_path):
    installer = tmp_path / 'SARU-Picker-Setup-3.0.0-win64.exe'
    with installer.open('wb') as target:
        target.truncate(downloads.INSTALLER_MAX_BYTES)
    with pytest.raises(ValueError, match='installer alone exceeds'):
        downloads.prepare(tmp_path)
    assert not (tmp_path / 'download-manifest.json').exists()
