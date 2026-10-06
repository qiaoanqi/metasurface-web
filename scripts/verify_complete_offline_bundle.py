"""Verify the ZIP and installed release files without modifying the package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def verify(archive_path: Path, installed: Path) -> dict:
    with ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
        manifest_names = [n for n in archive.namelist() if n.count('/') == 1 and n.endswith('/RELEASE_MANIFEST.json')]
        if len(manifest_names) != 1:
            raise ValueError("Expected a single package-root manifest")
        prefix = manifest_names[0].rsplit('/', 1)[0] + '/'
        manifest = json.loads(archive.read(manifest_names[0]))
        for relative, expected in manifest['files'].items():
            zipped = archive.read(prefix + relative)
            local = (installed / relative).read_bytes()
            if 'sha256:' + hashlib.sha256(zipped).hexdigest().upper() != expected:
                raise ValueError(f"ZIP hash mismatch: {relative}")
            if local != zipped:
                raise ValueError(f"Installed file differs: {relative}")
        if (installed / 'RELEASE_MANIFEST.json').read_bytes() != archive.read(manifest_names[0]):
            raise ValueError("Installed manifest differs")
        if len(archive.namelist()) != len(manifest['files']) + 1:
            raise ValueError("ZIP inventory differs from manifest")
        for relative in manifest['entrypoints'].values():
            if relative not in manifest['files']:
                raise ValueError(f"Unregistered entry point: {relative}")
        for name in ('启动离线演示.bat', '停止离线演示.bat'):
            content = archive.read(prefix + name)
            content.decode('ascii')
            if not content.startswith(b'@echo off\r\n'):
                raise ValueError(f"Unsupported batch encoding: {name}")
        required = (
            'runtime/competition/reference_library.py',
            'runtime/competition/tio2_air_reference_records_v1.jsonl',
            'runtime/competition/tio2_air_day_audit_20260930.json',
            'runtime/competition/tio2_air_day_color_audit_20260930.json',
            'common_offline.ps1', 'start_offline.ps1', 'stop_offline.ps1',
        )
        for name in required:
            if name not in manifest['files']:
                raise ValueError(f"Missing runtime asset: {name}")
        return {'status': 'pass', 'files_verified': len(manifest['files']),
                'zip_entries': len(archive.namelist()), 'installed': str(installed),
                'zip_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest().upper()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--installed', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.installed), ensure_ascii=False))
