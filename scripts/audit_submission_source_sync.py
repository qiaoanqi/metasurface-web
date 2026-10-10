"""Compare every submission file with its authoritative source or adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

if __package__:
    from . import build_complete_offline_bundle as offline
else:
    import build_complete_offline_bundle as offline


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PREFIX = '04_可运行软件/' + offline.PACKAGE_DIR_NAME + '/'
MATERIALS = Path('competition/submission_materials_20261010')
EVIDENCE = {
    'tio2_air_day_audit_20260930.json': 'competition/tio2_air_day_audit_20260930.json',
    'tio2_air_day_color_audit_20260930.json': 'competition/tio2_air_day_color_audit_20260930.json',
    'offline_full_feature_review_20261006.json': '.state/offline_full_feature_review_20261006.json',
    'offline_feature_runtime_audit_final_20261006.json': '.state/offline_feature_runtime_audit_final_20261006.json',
    'offline_real_app_flows_final_20261006.json': '.state/offline_real_app_flows_final_20261006.json',
    'offline_flow_smart_20261006.json': '.state/offline_flow_smart_20261006.json',
    'offline_smart_grid_verified_20261006.png': '.state/ui_previews/offline_smart_grid_verified_20261006.png',
    'offline_light_pattern_verified_20261006.png': '.state/ui_previews/offline_light_pattern_verified_20261006.png',
}


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


def generated_text(value: str, *, bom: bool = False, ascii_only: bool = False) -> bytes:
    """Match Path.write_text on the Windows build host (CRLF + optional BOM)."""
    normalized = value.replace('\r\n', '\n').replace('\n', '\r\n')
    encoding = 'ascii' if ascii_only else ('utf-8-sig' if bom else 'utf-8')
    return normalized.encode(encoding)


def expected_source(relative: str, root: Path) -> tuple[str, str, bytes | None]:
    """Resolve only the public runtime inventory, not the research directory."""
    if '\\' in relative or relative.startswith('/') or '..' in relative.split('/'):
        raise ValueError(f'Unsafe source path: {relative}')
    source = None
    kind = 'direct'
    if relative.startswith(RUNTIME_PREFIX):
        local = relative[len(RUNTIME_PREFIX):]
        generated = {
            'README_先看这里.md': generated_text(offline.README),
            'audit/README_证据索引.md': generated_text(offline.AUDIT_README),
            'docs/11_本地离线演示说明.md': generated_text(offline.OFFLINE_GUIDE),
            'logs/README.md': generated_text(offline.LOG_README),
            'common_offline.ps1': generated_text(offline.COMMON_PS1, bom=True),
            'start_offline.ps1': generated_text(offline.START_PS1, bom=True),
            '启动离线演示.bat': generated_text(offline.START_BAT, ascii_only=True),
            'runtime/requirements-offline.txt': generated_text(offline.OFFLINE_REQUIREMENTS),
        }
        if local in generated:
            return 'generated', 'scripts/build_complete_offline_bundle.py:' + local, generated[local]
        if local in ('RELEASE_MANIFEST.json', 'audit/ui-release-manifest.json'):
            return 'generated_manifest', local, None
        if local == 'showcase/index.html':
            source = offline.SHOWCASE_SOURCE / 'index.html'
            value = offline.render_offline_showcase((root / source).read_text(encoding='utf-8'))
            return 'offline_adapter', source.as_posix(), generated_text(value)
        if local == 'desktop_host.py':
            source = Path('scripts/offline_desktop_host.py')
        elif local.startswith('runtime/'):
            source = Path(local[len('runtime/'):])
        elif local.startswith(('deployment/', 'protocols/')):
            source = Path(local)
        elif local.startswith('showcase/'):
            source = offline.SHOWCASE_SOURCE / local[len('showcase/'):]
        elif local.startswith('docs/'):
            source = Path('competition') / local[len('docs/'):]
        elif local.startswith('audit/') and Path(local).name in EVIDENCE:
            source = Path(EVIDENCE[Path(local).name])
        elif local == 'audit/RELEASE_MANIFEST_v3.json':
            source = Path('_archive/legacy_web_releases_20261006/AI超表面结构色智能设计系统_发布包_v3/RELEASE_MANIFEST.json')
            kind = 'pinned_historical_evidence'
    elif relative == 'README_提交包先看.md' or relative.startswith(('02_软件与算法/', '06_来源与授权/')):
        source = MATERIALS / relative
    elif relative.startswith('03_测试与审计/evidence/'):
        source = Path(EVIDENCE[Path(relative).name]) if Path(relative).name in EVIDENCE else None
        kind = 'dated_evidence'
    elif relative.startswith('03_测试与审计/') or relative == '01_作品材料/01_报名简介_最终版.md':
        source = MATERIALS / relative
    elif relative == '01_作品材料/02_作品简介_最终版_v1.pdf':
        source = Path('competition/作品简介_最终版_v1.pdf')
    elif relative == '01_作品材料/03_技术方案_最终版_v1.pdf':
        source = Path('competition/技术方案_最终版_v1.pdf')
    else:
        match = re.fullmatch(r'01_作品材料/(?:04_答辩PPT_匿名公开_(v\d+)\.pptx|05_答辩PPT_匿名公开_(v\d+)\.pdf)', relative)
        if match:
            version = match[1] or match[2]
            source = Path('competition/答辩材料_20261009') / f'AI超表面结构色智能设计系统_答辩修订版_{version}{Path(relative).suffix}'
        match = re.fullmatch(r'05_演示视频/AI超表面结构色智能设计系统_演示视频_(v\d+)(\.mp4|_验证回执\.json)', relative)
        if match:
            source = Path('competition') / (
                f'AI超表面结构色智能设计系统_自然旁白_AI配乐_{match[1]}.mp4' if match[2] == '.mp4'
                else f'AI超表面结构色智能设计系统_答辩视频_{match[1]}_verification-receipt.json'
            )
    if source is None:
        raise ValueError(f'Unmapped submission file: {relative}')
    return kind, source.as_posix(), (root / source).read_bytes()


def compare_file(relative: str, payload: bytes, root: Path) -> dict:
    result = {'path': relative, 'packaged_sha256': digest(payload)}
    try:
        kind, source, expected = expected_source(relative, root)
        result.update(kind=kind, source=source)
        if expected is None:
            value = json.loads(payload)
            if value.get('status') != 'pass' or not value.get('files'):
                raise ValueError('Invalid generated manifest')
            if relative.endswith('/audit/ui-release-manifest.json'):
                if value.get('root') != 'runtime':
                    raise ValueError('UI manifest must describe shipped runtime')
                for local, sha in value['files'].items():
                    actual = digest((root / local).read_bytes())
                    if sha.upper() != 'SHA256:' + actual:
                        raise ValueError(f'UI source drift: {local}')
            result['status'] = 'pass'
        else:
            result['source_sha256'] = digest(expected)
            result['status'] = 'pass' if payload == expected else 'mismatch'
    except (OSError, ValueError, KeyError) as exc:
        result.update(status='error', error=str(exc))
    return result


def latest_versions(root: Path, files: dict) -> dict:
    patterns = {
        'pptx': ('competition/答辩材料_20261009', 'AI超表面结构色智能设计系统_答辩修订版_v*.pptx', '01_作品材料/04_'),
        'ppt_pdf': ('competition/答辩材料_20261009', 'AI超表面结构色智能设计系统_答辩修订版_v*.pdf', '01_作品材料/05_'),
        'video': ('competition', 'AI超表面结构色智能设计系统_自然旁白_AI配乐_v*.mp4', '05_演示视频/'),
    }
    results = {}
    for key, (directory, pattern, prefix) in patterns.items():
        candidates = [(int(re.search(r'_v(\d+)\.', p.name)[1]), p) for p in (root / directory).glob(pattern)]
        if not candidates:
            results[key] = {'status': 'error', 'error': 'No source version found'}
            continue
        number, source = max(candidates)
        selected = [name for name in files if name.startswith(prefix) and Path(name).suffix == source.suffix]
        passed = len(selected) == 1 and re.search(r'_v(\d+)\.', selected[0]) and int(re.search(r'_v(\d+)\.', selected[0])[1]) == number
        results[key] = {'status': 'pass' if passed else 'mismatch', 'latest_source': source.relative_to(root).as_posix(), 'selected': selected}
    return results


def make_report(root: Path, files: dict, read) -> dict:
    entries = [compare_file(relative, read(relative), root) for relative in sorted(files)]
    versions = latest_versions(root, files)
    failures = [item for item in entries if item['status'] != 'pass']
    return {
        'schema': 'submission-source-sync-v1',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'local_submission_sources_only',
        'status': 'pass' if not failures and all(v['status'] == 'pass' for v in versions.values()) else 'fail',
        'files_checked': len(entries), 'failures': failures, 'latest_versions': versions,
        'classification_counts': {kind: sum(e.get('kind') == kind for e in entries) for kind in sorted({e.get('kind', 'unmapped') for e in entries})},
        'boundary': 'Historical audits are byte-verified evidence, not fresh functional tests. Research and cloud assets are out of scope.',
        'files': entries,
    }


def audit_directory(staging: Path, root: Path = ROOT) -> dict:
    manifest = json.loads((staging / 'SUBMISSION_MANIFEST.json').read_text(encoding='utf-8'))
    return make_report(root, manifest['files'], lambda relative: (staging / relative).read_bytes())


def audit_archive(archive_path: Path, root: Path = ROOT, installed: Path | None = None) -> dict:
    if __package__:
        from .build_final_competition_submission import verify_submission_archive
    else:
        from build_final_competition_submission import verify_submission_archive
    verification = verify_submission_archive(archive_path)
    with ZipFile(archive_path) as archive:
        name = next(n for n in archive.namelist() if n.count('/') == 1 and n.endswith('/SUBMISSION_MANIFEST.json'))
        prefix = name.rsplit('/', 1)[0] + '/'
        manifest = json.loads(archive.read(name))
        report = make_report(root, manifest['files'], lambda relative: archive.read(prefix + relative))
        if installed is not None:
            mismatches = [relative for relative in [*manifest['files'], 'SUBMISSION_MANIFEST.json']
                          if not (installed / relative).is_file() or (installed / relative).read_bytes() != archive.read(prefix + relative)]
            report['extracted_files'] = {'files_checked': len(manifest['files']) + 1, 'mismatches': mismatches, 'status': 'pass' if not mismatches else 'fail'}
            if mismatches:
                report['status'] = 'fail'
    report['archive'] = {'path': str(archive_path.resolve()), **verification}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--installed', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_archive(args.archive, args.root.resolve(), args.installed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
