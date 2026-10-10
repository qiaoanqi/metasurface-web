"""Build the single ZIP submitted for the AI+software innovation track.

The package intentionally contains the audited offline runtime as an unpacked
directory so a reviewer can inspect the source and run the Windows entrypoint
after extracting one archive.  It does not touch cloud or GitHub state.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = ROOT / "dist" / "AI超表面结构色智能设计系统_完整离线包_v1"
MATERIAL_SOURCE = ROOT / "competition" / "submission_materials_20261010"
STAGING = ROOT / "dist" / "AI超表面结构色智能设计系统_算法创新赛_AI+软件创新_404 Not Found队_最终提交包_20261010_v26"
OUTPUT = ROOT / "dist" / "3_算法创新赛_AI+软件创新_404 Not Found队_v26.zip"
PACKAGE_NAME = STAGING.name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, target, dirs_exist_ok=True)


def write_utf8(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_runtime_manifest(runtime: Path) -> str:
    manifest_path = runtime / "RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {}
    for path in sorted(runtime.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative = path.relative_to(runtime).as_posix()
        if ".venv" in path.relative_to(runtime).parts or "__pycache__" in path.relative_to(runtime).parts:
            continue
        if relative.startswith("logs/") and relative != "logs/README.md":
            continue
        files[relative] = "sha256:" + sha256(path)
    manifest["status"] = "pass"
    manifest["files"] = files
    manifest["file_count"] = len(files)
    write_utf8(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return sha256(manifest_path)


def add_submission_files() -> None:
    # Public materials only. Internal rehearsal notes and identity-bearing drafts
    # stay outside this directory.
    copy(MATERIAL_SOURCE / "README_提交包先看.md", STAGING / "README_提交包先看.md")
    copy(MATERIAL_SOURCE / "01_作品材料" / "01_报名简介_最终版.md", STAGING / "01_作品材料" / "01_报名简介_最终版.md")
    copy(ROOT / "competition" / "作品简介_最终版_v1.pdf", STAGING / "01_作品材料" / "02_作品简介_最终版_v1.pdf")
    copy(ROOT / "competition" / "技术方案_最终版_v1.pdf", STAGING / "01_作品材料" / "03_技术方案_最终版_v1.pdf")
    copy(ROOT / "competition" / "答辩材料_20261009" / "AI超表面结构色智能设计系统_答辩修订版_v23.pptx", STAGING / "01_作品材料" / "04_答辩PPT_匿名公开_v23.pptx")
    copy(ROOT / "competition" / "答辩材料_20261009" / "AI超表面结构色智能设计系统_答辩修订版_v23.pdf", STAGING / "01_作品材料" / "05_答辩PPT_匿名公开_v23.pdf")

    for path in (MATERIAL_SOURCE / "02_软件与算法").glob("*"):
        copy(path, STAGING / "02_软件与算法" / path.name)
    for path in (MATERIAL_SOURCE / "03_测试与审计").glob("*"):
        copy(path, STAGING / "03_测试与审计" / path.name)
    for path in (MATERIAL_SOURCE / "06_来源与授权").glob("*"):
        copy(path, STAGING / "06_来源与授权" / path.name)

    # Audits and reproducibility receipts are included separately from source code.
    audit_files = {
        ROOT / "competition" / "tio2_air_day_audit_20260930.json": "tio2_air_day_audit_20260930.json",
        ROOT / "competition" / "tio2_air_day_color_audit_20260930.json": "tio2_air_day_color_audit_20260930.json",
        ROOT / ".state" / "offline_full_feature_review_20261006.json": "offline_full_feature_review_20261006.json",
        ROOT / ".state" / "offline_feature_runtime_audit_final_20261006.json": "offline_feature_runtime_audit_final_20261006.json",
        ROOT / ".state" / "offline_real_app_flows_final_20261006.json": "offline_real_app_flows_final_20261006.json",
        ROOT / ".state" / "offline_flow_smart_20261006.json": "offline_flow_smart_20261006.json",
        ROOT / ".state" / "ui_previews" / "offline_smart_grid_verified_20261006.png": "offline_smart_grid_verified_20261006.png",
        ROOT / ".state" / "ui_previews" / "offline_light_pattern_verified_20261006.png": "offline_light_pattern_verified_20261006.png",
    }
    for source, name in audit_files.items():
        copy(source, STAGING / "03_测试与审计" / "evidence" / name)

    copy(ROOT / "competition" / "AI超表面结构色智能设计系统_自然旁白_AI配乐_v4.mp4", STAGING / "05_演示视频" / "AI超表面结构色智能设计系统_演示视频_v4.mp4")
    copy(ROOT / "competition" / "AI超表面结构色智能设计系统_答辩视频_v4_verification-receipt.json", STAGING / "05_演示视频" / "AI超表面结构色智能设计系统_演示视频_v4_验证回执.json")


def write_submission_manifest(runtime_manifest_sha256: str) -> None:
    files = {}
    manifest_path = STAGING / "SUBMISSION_MANIFEST.json"
    for path in sorted(STAGING.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        files[path.relative_to(STAGING).as_posix()] = "sha256:" + sha256(path)
    manifest = {
        "schema": "ai-metasurface-competition-submission-v1",
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track": "算法创新赛 · AI+软件创新",
        "team": "404 Not Found队",
        "registration_number": 3,
        "public_materials_anonymous": True,
        "cloud_updated": False,
        "github_updated": False,
        "runtime_release_manifest_sha256": runtime_manifest_sha256,
        "required_items": {
            "software_description": "02_软件与算法/软件说明与核心算法说明_最终版.md",
            "test_report": "03_测试与审计/测试报告_最终版.md",
            "pptx": "01_作品材料/04_答辩PPT_匿名公开_v23.pptx",
            "ppt_pdf": "01_作品材料/05_答辩PPT_匿名公开_v23.pdf",
            "demo_video": "05_演示视频/AI超表面结构色智能设计系统_演示视频_v4.mp4",
            "offline_entrypoint": "04_可运行软件/AI超表面结构色智能设计系统_完整离线包_v1/启动离线演示.bat",
        },
        "file_count": len(files),
        "files": files,
    }
    write_utf8(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> int:
    if not RUNTIME_SOURCE.is_dir():
        raise FileNotFoundError(RUNTIME_SOURCE)
    if STAGING.exists() or OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing final package: {STAGING} or {OUTPUT}")

    STAGING.mkdir(parents=True)
    runtime = STAGING / "04_可运行软件" / RUNTIME_SOURCE.name
    copy_tree(RUNTIME_SOURCE, runtime)
    runtime_manifest_sha256 = refresh_runtime_manifest(runtime)
    add_submission_files()
    write_submission_manifest(runtime_manifest_sha256)

    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(STAGING.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(STAGING).as_posix()
            info = ZipInfo(f"{PACKAGE_NAME}/{relative}", date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED)
    print(json.dumps({
        "staging": str(STAGING),
        "output": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "files": len(json.loads((STAGING / "SUBMISSION_MANIFEST.json").read_text(encoding="utf-8"))["files"]),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
