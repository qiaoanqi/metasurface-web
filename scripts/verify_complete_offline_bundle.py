"""Verify the ZIP and installed release files without modifying the package."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def registered_rcwa_weight_pairs(runtime: Path) -> tuple[tuple[str, str], ...]:
    """Derive inference/gradient asset pairs from the shipped registry."""
    tree = ast.parse((runtime / 'ml_module.py').read_text(encoding='utf-8-sig'))
    registries = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('_RCWA_MODELS', '_RCWA_SUBSTRATE_MODELS'):
                    registries[target.id] = ast.literal_eval(node.value)
    if set(registries) != {'_RCWA_MODELS', '_RCWA_SUBSTRATE_MODELS'}:
        raise ValueError('Missing registered RCWA model families')
    pairs = set()
    model_root = (runtime / 'models').resolve()
    for registry in registries.values():
        for patterns in registry.values():
            matches = sorted({path for pattern in patterns for path in model_root.glob(pattern)})
            if not matches:
                raise ValueError(f'Missing registered RCWA ONNX family: {patterns}')
            for path in matches:
                path.resolve().relative_to(model_root)
                relative = path.relative_to(runtime.resolve()).as_posix()
                pairs.add((relative, path.with_suffix('.pt').relative_to(runtime.resolve()).as_posix()))
    return tuple(sorted(pairs))


def validate_analysis_runtime(runtime: Path) -> dict:
    """Check shipped analysis dependencies without importing or running the app."""
    runtime = runtime.resolve()
    files = {}

    def read(relative: str) -> bytes:
        path = (runtime / relative).resolve()
        path.relative_to(runtime)
        if not path.is_file():
            raise ValueError(f'Missing analysis dependency: {relative}')
        content = path.read_bytes()
        files[relative] = 'sha256:' + hashlib.sha256(content).hexdigest().upper()
        return content

    def assignment(relative: str, name: str):
        tree = ast.parse(read(relative).decode('utf-8-sig'))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return node.value
        raise ValueError(f'Missing analysis contract: {relative}:{name}')

    sources = ast.literal_eval(assignment('app.py', '_ANALYSIS_SOURCE_DEPENDENCIES'))
    for relative in sorted(set((
        'ui_analysis_snapshots.py', 'ui_model_resources.py',
        *(relative for route in sources.values() for relative in route),
    ))):
        read(relative)

    contract_node = assignment('ui_model_difference_contracts.py', 'GENERIC_ONNX_ROUTE')
    if not isinstance(contract_node, ast.Call):
        raise ValueError('Unsupported generic ONNX contract')
    contract = {
        item.arg: ast.literal_eval(item.value)
        for item in contract_node.keywords
        if item.arg and item.arg.endswith(('_relative_path', '_sha256'))
    }
    for path_key, hash_key in (
        ('model_relative_path', 'model_sha256'),
        ('external_data_relative_path', 'external_data_sha256'),
        ('source_pt_relative_path', 'source_pt_sha256'),
    ):
        relative = contract[path_key]
        if hashlib.sha256(read(relative)).hexdigest() != contract[hash_key]:
            raise ValueError(f'Analysis artifact hash mismatch: {relative}')

    evidence = {}
    for key in ('conversion_protocol', 'conversion_result'):
        relative = contract[key + '_relative_path']
        value = json.loads(read(relative))
        canonical = (json.dumps(value, ensure_ascii=True, sort_keys=True,
                                separators=(',', ':')) + '\n').encode('utf-8')
        if hashlib.sha256(canonical).hexdigest() != contract[key + '_sha256']:
            raise ValueError(f'Analysis evidence hash mismatch: {relative}')
        evidence[key] = value
    result = evidence['conversion_result']
    if (result.get('passed') is not True or
            result.get('protocol_sha256') != contract['conversion_protocol_sha256']):
        raise ValueError('Conversion evidence did not pass or bind the shipped protocol')
    script = result['artifacts']['audit_script']
    if hashlib.sha256(read(script['path'])).hexdigest() != script['sha256']:
        raise ValueError(f'Analysis audit script hash mismatch: {script["path"]}')
    figure_path = ast.literal_eval(assignment('ui_fdtd_asset.py', 'FDTD_ASSET_RELATIVE_PATH'))
    figure_hash = ast.literal_eval(assignment('ui_fdtd_asset.py', 'FDTD_ASSET_SHA256'))
    if hashlib.sha256(read(figure_path)).hexdigest().upper() != figure_hash.upper():
        raise ValueError(f'Historical figure hash mismatch: {figure_path}')
    for onnx_relative, pt_relative in registered_rcwa_weight_pairs(runtime):
        read(onnx_relative)
        read(pt_relative)
    for relative in ('rl_design.py', 'models/rl_qtable.npy', 'models/rl_qtable_meta.npy'):
        read(relative)
    return {'status': 'pass', 'files': dict(sorted(files.items()))}


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
        if 'stop' in manifest['entrypoints']:
            raise ValueError('Obsolete stop entry point is still registered')
        for name in ('停止离线演示.bat', 'stop_offline.ps1'):
            if prefix + name in archive.namelist() or (installed / name).exists():
                raise ValueError(f'Obsolete stop file still exists: {name}')
        for name in ('启动离线演示.bat',):
            content = archive.read(prefix + name)
            content.decode('ascii')
            if not content.startswith(b'@echo off\r\n'):
                raise ValueError(f"Unsupported batch encoding: {name}")
        required = (
            'runtime/competition/reference_library.py',
            'runtime/competition/tio2_air_reference_records_v1.jsonl',
            'runtime/competition/tio2_air_day_audit_20260930.json',
            'runtime/competition/tio2_air_day_color_audit_20260930.json',
            'common_offline.ps1', 'start_offline.ps1',
            'desktop_host.py',
        )
        for name in required:
            if name not in manifest['files']:
                raise ValueError(f"Missing runtime asset: {name}")
        analysis = validate_analysis_runtime(installed / 'runtime')
        for relative, expected in analysis['files'].items():
            if manifest['files'].get('runtime/' + relative) != expected:
                raise ValueError(f'Unregistered analysis dependency: {relative}')
        ui_manifest = json.loads(archive.read(prefix + 'audit/ui-release-manifest.json'))
        if ui_manifest.get('root') != 'runtime':
            raise ValueError('UI manifest is not bound to the shipped runtime')
        for relative, expected in ui_manifest['files'].items():
            content = archive.read(prefix + 'runtime/' + relative)
            if hashlib.sha256(content).hexdigest().upper() != expected.split(':', 1)[1].upper():
                raise ValueError(f'UI runtime manifest hash mismatch: {relative}')
        return {'status': 'pass', 'analysis_dependencies_verified': len(analysis['files']),
                'files_verified': len(manifest['files']),
                'zip_entries': len(archive.namelist()), 'installed': str(installed),
                'zip_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest().upper()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--installed', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.installed), ensure_ascii=False))
